import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import csv

def xlnet(pos_file_path, neg_file_path, test_file_path, load_model=True, load_model_path='../data/XLNet/model.pth', num_epoch=0, batch_size=64, max_token_length=64, lr=2e-5):
    pos_input_file_path = pos_file_path
    neg_input_file_path = neg_file_path
    test_input_file_path = test_file_path
    
    # Read data
    pos_list = []
    with open(pos_input_file_path, encoding='UTF-8') as f:
        for line in f:
            pos_list.append(line.strip())
    data_pos = pd.DataFrame(pos_list)
    
    neg_list = []
    with open(neg_input_file_path, encoding='UTF-8') as f:
        for line in f:
            neg_list.append(line.strip())
    data_neg = pd.DataFrame(neg_list)

    data_pos['label'] = 1
    data_neg['label'] = 0
    
    # Combine and shuffle data
    data = pd.concat([data_neg,data_pos], axis=0)
    data = data.sample(frac=1).reset_index()
    data = data.rename(columns = {0:'sentence'})
    data = data.drop(['index'],axis=1)

    # Fetch inputs and labels
    sentences = data.sentence.values
    labels = data.label.values

    # garbage collection
    del pos_list, data_pos
    del neg_list, data_neg
    del data
    gc.collect()

    # Tokenize inputs
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print('Size of input samples: ', len(tokenized_texts))

    # Garbage collection
    del sentences
    gc.collect()
    
    # Convert word to id and pad
    MAX_LEN = max_token_length
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    
    # Garbage collection
    del tokenizer
    del tokenized_texts
    gc.collect()
    
    # Split dataset
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2021, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2021, test_size=0.1)
    print('Size of training samples: ', len(train_inputs))
    print('Size of validation samples: ', len(validation_inputs))
    
    # Convert tokenized inputs to tensor
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Garbage collection
    del input_ids
    del attention_masks
    del labels
    gc.collect()

    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128
    # batch_size = 64

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Garbage collection
    del train_inputs, train_masks, train_labels, train_data, train_sampler
    del validation_inputs, validation_masks, validation_labels, validation_data, validation_sampler
    gc.collect()

    # Process test data
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    test_data = []
    with open(test_input_file_path, encoding='UTF-8') as f:
        for line in f:
            test_data.append(line.strip())

    test_data = pd.DataFrame(test_data)
    test_data = test_data.rename(columns = {0:'sentence'})
    test_data['label'] = 1

    sentences = test_data.sentence.values
    labels = test_data.label.values
    
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print('Size of test samples: ', len(tokenized_texts))
    
    MAX_LEN = max_token_length
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    # batch_size = 64

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Garbage collection
    del test_data
    del tokenized_texts
    del sentences
    del labels
    del input_ids
    del attention_masks
    del prediction_inputs, prediction_masks, prediction_labels, prediction_data, prediction_sampler
    gc.collect()
    
    # Function to write prediction file
    import csv
    def create_csv_submission(ids, y_pred, name):
        """
        Creates an output file in .csv format for submission to Kaggle or AIcrowd
        Arguments: ids (event ids associated with each prediction)
                   y_pred (predicted class labels)
                   name (string name of .csv output file to be created)
        """
        with open(name, 'w') as csvfile:
            fieldnames = ['Id', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for r1, r2 in zip(ids, y_pred):
                writer.writerow({'Id':int(r1),'Prediction':int(r2)})

    # Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top. 
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
    model.to(device)
    if load_model:
        print('Loading trained model...')
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        print('Finish loading')

    # Optimizer and parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Number of training epochs (authors recommend between 2 and 4)
    EPOCHS = num_epoch
    
    # trange is a tqdm wrapper around the normal python range
    for epoch in range(1, EPOCHS + 1):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        pbar = tqdm(train_dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))

        # Train the data for one epoch
        for batch in pbar:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        torch.save(model.state_dict(), '../data/XLNet/model.pth')

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
              # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        # Prediction on test set

        # Put model in evaluation mode
        model.eval()

    # Tracking variables 
    predictions = []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Store predictions and true labels
        predictions += logits.tolist()

    pred_flat = np.argmax(predictions, axis=1).flatten()
    pred_flat = np.array(pred_flat)*2-1
    pred_flat = list(pred_flat)
    create_csv_submission(np.arange(1, len(pred_flat)+1), pred_flat, '../output/submission.csv')


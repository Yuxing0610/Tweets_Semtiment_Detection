import pandas as pd
import os
import torch
from transformers import BertTokenizer, AutoTokenizer
import sys
import random
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoModel
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime

# If use self-pretrained model, must use GPU devices.
# Usage: BERT(model_load_flag=False, model_type='bert')
def BERT(pos_file_path, neg_file_path, test_file_path, load_bert_model_dir, batch_size=32, epochs=4, model_load_flag=False, model_type='bertweet'):
    """""
    params: model_load_flag: use self-pretrained model or initial pretained model
            model_type: 'bert', 'bertweet'
    """""
    #config
    output_dir = '../output/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Tokenizer select
    if model_type == 'bert':
        if model_load_flag:
            tokenizer = BertTokenizer.from_pretrained(load_bert_model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif model_type == 'bertweet':
        if model_load_flag:
            tokenizer = AutoTokenizer.from_pretrained(load_bert_model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=False)
    else:
        print('Select correct model !')
        sys.exit(1)
    
    # Load train data
    pos_list = []
    with open(pos_file_path,encoding='UTF-8') as f:
        for line in f:
            pos_list.append(line.strip())
    data_pos = pd.DataFrame(pos_list)
    
    neg_list = []
    with open(neg_file_path,encoding='UTF-8') as f:
        for line in f:
            neg_list.append(line.strip())
    data_neg = pd.DataFrame(neg_list)

    data_pos['label'] = 1
    data_neg['label'] = 0
    
    # Combine and shuffle data
    traindata = pd.concat([data_neg,data_pos], axis=0)
    traindata = traindata.sample(frac=1).reset_index()
    traindata = traindata.rename(columns = {0:'tweet'})
    traindata = traindata.drop(['index'],axis=1)
    sentences = traindata.tweet.values
    labels = traindata.label.values

    # Load test data
    testdata = []
    with open(test_file_path,encoding='UTF-8') as f:
        for line in f:
            testdata.append(line.strip())

    testdata = pd.DataFrame(testdata)
    testdata = testdata.rename(columns = {0:'tweet'})
    testdata['label'] = 1

    # Test
    print(' Original: ', sentences[0])
    print('Tokenized: ', tokenizer.tokenize(sentences[0]))
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
    print(sentences.shape)

    # Tokenize all of the sentences and map the tokens to word IDs.
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True, # maybe not needed
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # DataLoaders
    train_dataloader = DataLoader(
                train_dataset, 
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size
            )

    # select model
    class BertClassifier(nn.Module):
        def __init__(self, load_flag):
            super(BertClassifier, self).__init__()
            # Specify hidden size of BERT, hidden size of our classifier, and number of labels
            D_in, H, D_out = 768, 50, 2
            # Load BERT
            if model_type == 'bert':
                if load_flag:
                    self.bert = BertModel.from_pretrained(load_bert_model_dir)
                else:
                    self.bert = BertModel.from_pretrained('bert-base-uncased')
            elif model_type == 'bertweet':
                if load_flag:
                    self.bert = AutoModel.from_pretrained(load_bert_model_dir)
                else:
                    self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
            else:
                print('Select correct model !')
                sys.exit(1)

            # Instantiate an one-layer FNN classifier
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Linear(H, D_out)
            )

            # load_model_param
            if load_flag:
                if model_type == 'bert':
                    self.classifier.load_state_dict(torch.load(load_bert_model_dir+'classifier.pth'))
                elif model_type == 'bertweet':
                    self.classifier.load_state_dict(torch.load(load_bert_model_dir+'classifier.pth'))
            
        def forward(self, input_ids, attention_mask):
            # Feed input to BERT
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            
            # Add a FNN layer to BERT
            last_hidden_state_cls = outputs[0][:, 0, :]
            logits = self.classifier(last_hidden_state_cls)

            return logits

    model = BertClassifier(load_flag=model_load_flag)
    model.to(device)

    # Train setting
    total_steps = len(train_dataloader) * epochs
    # Optimizer
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # learning_rate
                    eps = 1e-8 # adam_epsilon
    )

    # Learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Convert time in seconds to hh:mm:ss
    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    #Training

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_func = nn.CrossEntropyLoss()

    training_stats = []
    total_t0 = time.time()
    save_loss = 1000

    for epoch_i in range(0, epochs):
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        t0 = time.time()
        total_train_loss = 0
        model.train()

        #####Train#####
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Time: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            # clear gradient calculated before
            model.zero_grad()        

            # forward pass
            logits = model(b_input_ids, b_input_mask)
            loss = loss_func(logits,b_labels)
            total_train_loss += loss.item()

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        #####Validation#####
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                logits = model(b_input_ids,
                            b_input_mask)
            loss = loss_func(logits,b_labels)

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        #accuracy & loss output
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        # Record
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
            }
        )
        
        # Save model
        if avg_val_loss < save_loss:
            save_loss = avg_val_loss
            model.bert.save_pretrained(output_dir)
            torch.save(model.classifier.state_dict(), output_dir+'classifier.pth')
            tokenizer.save_pretrained(output_dir)

    print("Total training time {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # Load test dataset

    print('Number of test sentences: {:,}\n'.format(testdata.shape[0]))

    sentences = testdata.tweet.values
    labels = testdata.label.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                     
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',    
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction
    model.eval()

    predictions = []
    for batch in prediction_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids,
                        attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    # Output submit file
    import csv
    pred=[]

    for i in range(len(predictions)):
        for v in np.argmax(predictions[i], axis=1).flatten():
            pred.append(v)

    y_pred = np.array(pred)*2-1

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

    create_csv_submission(np.arange(1, len(y_pred)+1), y_pred, output_dir+'submission.csv')

    return 0

# Debug
#pos_input_file_path = '../twitter-datasets/train_pos.txt'
#neg_input_file_path = '../twitter-datasets/train_neg.txt'
#test_input_file_path = '../twitter-datasets/test_data.txt'
#BERT(pos_input_file_path,neg_input_file_path,test_input_file_path,'../data/model/',8,4,False,'bert')
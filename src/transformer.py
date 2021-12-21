import pandas as pd
import numpy as np
import torch
import csv
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from collections import Counter, OrderedDict
from torchtext.vocab import vocab, GloVe
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def Transformer(pos_file_path, neg_file_path, test_file_path):

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
    TwitterData = pd.concat([data_neg,data_pos], axis=0)
    TwitterData = TwitterData.sample(frac=1).reset_index()
    TwitterData = TwitterData.rename(columns = {0:'tweet'})
    TwitterData = TwitterData.drop(['index'],axis=1)

    # Load test data
    test_data = []
    with open(test_file_path,encoding='UTF-8') as f:
        for line in f:
            test_data.append(line.strip())

    test_data = pd.DataFrame(test_data)
    test_data = test_data.rename(columns = {0:'tweet'})
    test_data['label'] = 1

    output_dir = './data/'
    train_size = int(0.9 * len(TwitterData))
    EPOCHS = 40  # change number of epochs and see differences in performance

    # Create dataset
    class TextClassificationDataset(Dataset):
        def __init__(self, dataset1, tokenizer, text_vocab, label_vocab, split='train'):
            print(f'Numericalising tokens for {split} set...', end="", flush=True)
            data = dataset1
            self.dset = []
            self.labels = []
            self.text_vocab = text_vocab
            self.label_vocab = label_vocab
            for tweet, label in data:
                tokens = tokenizer(tweet.rstrip())
                self.dset.append([self.text_vocab[w] for w in tokens])
                self.labels.append(self.label_vocab[str(label)])
            print(f'Number of {split} samples: {len(self.dset)}')

        def __len__(self):
            return len(self.dset)

        def __getitem__(self, idx):
            tokens = self.dset[idx]
            label = self.labels[idx]
            return tokens, label

        def tokens(self):
            return self.text_vocab.get_itos()

        def vocab_size(self):
            return len(self.text_vocab)

        def num_classes(self):
            return len(self.label_vocab)

        @classmethod
        def build_vocab(cls, dataset1, tokenizer, split='train', min_freq=1, pad_token='<pad>', unk_token='<unk>'):
            print(f'Building vocab for twitter...', end="", flush=True)
            data = dataset1
            tokens = []
            labels = []
            for tweet, label in data:
                tokens += tokenizer(tweet.rstrip())
                labels.append(str(label))

            def create_vocab(counts, mf=1):
                counter = Counter(counts)
                sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                ordered_dict = OrderedDict(sorted_by_freq_tuples)
                return vocab(ordered_dict, min_freq=mf)

            text_vocab = create_vocab(tokens, min_freq)
            # set index for padding token
            text_vocab.insert_token(pad_token, 0)
            text_vocab.append_token(unk_token)
            text_vocab.set_default_index(text_vocab[unk_token])
            label_vocab =  create_vocab(labels)
            print(f'Number of tokens: {len(text_vocab)}')
            print(f'Classes: {label_vocab.get_stoi()}')
            return cls(dataset1, tokenizer, text_vocab, label_vocab, split)
        
    seed = 42
    torch.manual_seed(seed)
    tokenizer = get_tokenizer("basic_english")

    TwitterTrain=TwitterData[0:train_size]
    TwitterTest=TwitterData[train_size:]
    train_set = TextClassificationDataset.build_vocab(TwitterTrain.values, tokenizer, split='train', min_freq=2)

    vec = GloVe(name='6B', dim=100)
    embeddings = vec.get_vecs_by_tokens(train_set.tokens(), lower_case_backup=True)

    class AttentionModel(nn.Module):
        def __init__(
                self,
                vocab_size,
                num_label,
                e_dim=100,
                num_layer=1,
                num_head=2,
                dropout=0.1,
                max_len=512,
                padding_idx=0,
                weights=None,
                freeze=True,
                device=torch.device('cuda:0')):

            super().__init__()
            self.padding_idx = padding_idx
            self.max_len = max_len
            if weights is None:
                self.word_embeddings = nn.Embedding(vocab_size, e_dim, padding_idx=padding_idx)
                if freeze:
                    self.word_embeddings.weight.requires_grad = False
            else:
                self.word_embeddings = nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=padding_idx)
            self.position_embeddings = nn.Embedding(max_len, e_dim)
            self.position_ids = torch.arange(max_len).to(device)
            self.LayerNorm = nn.LayerNorm(e_dim, eps=1e-5)
            self.dropout = nn.Dropout(dropout)

            transformer_layer = nn.TransformerEncoderLayer(d_model=e_dim, nhead=num_head, dim_feedforward=e_dim * 4)
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layer)
            self.classifier = nn.Linear(e_dim, num_label)
            self.device = device
            self.to(device)

        def forward(self, inputs):
            token_ids, attn_masks = inputs
            batch_size = token_ids.size(0)
            batch_max_length = token_ids.size(1)
            # compute embeddings
            token_embs = self.word_embeddings(token_ids)
            pos_embs = self.position_embeddings(self.position_ids[:batch_max_length])
            embeddings = token_embs + pos_embs
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            # compute contextualised embeddings with transformer
            contextualised_embs = self.transformer(embeddings.permute(1, 0, 2), src_key_padding_mask=attn_masks)
            outputs = contextualised_embs.mean(0)
            logits = self.classifier(outputs)
            return logits

        def knn(self, token_ids, k=10):
            query = self.word_embeddings.weight[token_ids]
            x_src = F.normalize(query)
            x_tgt = F.normalize(self.word_embeddings.weight)
            # compute cosine similarity
            scores = x_src @ x_tgt.t()
            top_values, top_indices = torch.topk(scores, k + 1)
            return top_indices[:, 1:]  # remove top1 since it is the target token

        def collate_batch(self, batch):
            label_list, text_list, lengths = [], [], []
            for (_text, _label) in batch:
                label_list.append(_label)
                text_list.append(_text)
                lengths.append(len(_text))
            max_length = min(max(lengths), self.max_len)
            # truncate or add padding to the right hand side
            for i, _text in enumerate(text_list):
                if len(_text) < max_length:  # pad
                    text_list[i] += [self.padding_idx] * (max_length - len(_text))
                else:  # truncate
                    text_list[i] = _text[:max_length]
            label_list = torch.tensor(label_list, dtype=torch.long).to(self.device)
            text_list = torch.tensor(text_list, dtype=torch.long).to(self.device)
            attn_mask = text_list == self.padding_idx
            return label_list, (text_list, attn_mask)

    model = AttentionModel(
        vocab_size=train_set.vocab_size(), 
        num_label=train_set.num_classes(), 
        e_dim=100,
        max_len=512,
        num_head=2,
        num_layer=1,
        weights=embeddings,
        freeze='store_true',
        padding_idx=train_set.text_vocab['<pad>'],
        device=torch.device('cuda:0'))

    @torch.no_grad()
    def word_knn(model, valid_set, vocab, top=10):
        print('--------------------------------------------------------------------------------')
        print('Validation - top 10 nearest neighbors ------------------------------------------')
        valid_token_ids = vocab.lookup_indices(valid_set)
        top_indices = model.knn(valid_token_ids, top)

        for i, word in enumerate(valid_set):
            results = ' '.join([vocab.lookup_token(top_indices[i, k].item()) for k in range(top)])
            print(f'{word}: {results}')
        print('--------------------------------------------------------------------------------')


    # define the loss function
    ce_loss = nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    steper = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    train_dataloader = DataLoader(train_set, 
                                batch_size=512, 
                                shuffle=True, 
                                collate_fn=model.collate_batch)


    # loop over training epochs
    min_loss = 1000
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        steper.step()
        for labels, inputs in pbar: 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            # display the loss 
            pbar.set_postfix(loss=loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(state_dict,output_dir+'model.pth')


    ##### Validation
    test_set = TextClassificationDataset(
        TwitterTest.values,
        tokenizer, 
        text_vocab=train_set.text_vocab, 
        label_vocab=train_set.label_vocab,
        split='test')

    test_dataloader = DataLoader(test_set, 
                                batch_size=512, 
                                shuffle=False, 
                                collate_fn=model.collate_batch)

    y_true = []
    y_pred = []

    with torch.no_grad():
        model.eval()
        for labels, inputs in tqdm(test_dataloader, desc='[Testing]'):
            logits = model(inputs)
            y_pred += logits.argmax(dim=1).tolist()
            y_true += labels.tolist()

    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print('------- Evaluation metrics --------')
    print(f'Precision: {pre*100:.2f}%')
    print(f'Recall: {rec*100:.2f}%')
    print(f'F1 score: {f1*100:.2f}%')
    print('-' * 35)

    ##### Testing
    submit_data = TextClassificationDataset(
        test_data.values,
        tokenizer, 
        text_vocab=train_set.text_vocab, 
        label_vocab=train_set.label_vocab,
        split='train')

    test_dataloader = DataLoader(submit_data, 
                                batch_size=512, 
                                shuffle=False, 
                                collate_fn=model.collate_batch)

    y_pred=[]
    with torch.no_grad():
        model.eval()
        for labels, inputs in tqdm(test_dataloader, desc='[Testing]'):
            logits = model(inputs)
            y_pred += logits.argmax(dim=1).tolist()

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

    y_pred = np.array(y_pred)*2-1
    y_pred = list(y_pred)
    create_csv_submission(np.arange(1, len(y_pred)+1), y_pred, output_dir+'submit.csv')

# Debug
#pos_input_file_path = '../twitter-datasets/train_pos.txt'
#neg_input_file_path = '../twitter-datasets/train_neg.txt'
#test_input_file_path = '../twitter-datasets/test_data.txt'
#Transformer(pos_input_file_path,neg_input_file_path,test_input_file_path)
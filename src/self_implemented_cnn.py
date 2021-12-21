import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import numpy as np
import random
from nltk.tokenize import TweetTokenizer, word_tokenize
import glove_embedding
import helpers

#define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   

        # first layer conv
        self.conv1 = nn.Sequential(
            # Input size :[1,50,200]
            nn.Conv2d(
                in_channels=1,    
                out_channels=16,  
                kernel_size=(5,1),    
                stride=1,         
                padding=(2,0)
            ),
            # Output size [16,50,200]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,1))   # Output size [16, 25, 200]
        )

        # Second Layer conv
        self.conv2 = nn.Sequential(
            #Input size [16, 25, 200]
            nn.Conv2d(
                in_channels=16,    
                out_channels=32,
                kernel_size=(5,1),
                stride=1,
                padding=(2,0)
            ),
            #Output size [32, 25, 200]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(5,1))  #Output size[32,5,200]
        )

        # Third Layer conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                #Input size[32, 5, 200]
                in_channels=32,    
                out_channels=32,
                kernel_size=(5,1),
                stride=1,
                padding=0
            ),
            # Output size [32,1,200]
            nn.ReLU(),
        )
        # deep fully connected network
        self.linear1 = nn.Linear(in_features=32*1*200, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear3 = nn.Linear(in_features=1000, out_features=1000)
        #Final layer to output the claasification result
        self.linear4 = nn.Linear(in_features=1000, out_features=2)
        #set dropout to avoid overfitting
        self.dropout = nn.Dropout(p=0.5) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #Strech the 2 dimensional tensor to 1 dimension for fully connected network
        x = x.view(x.size(0), -1)  

        x = F.relu(self.linear1(x)) 
        x = self.dropout(x)  
        x = F.relu(self.linear2(x)) 
        x = self.dropout(x) 
        x = F.relu(self.linear3(x)) 
        x = self.dropout(x) 
        x = self.linear4(x)   
        return x


def self_implemented_cnn(pos_file_path, neg_file_path, test_file_path, glove_method = "merged", parameter = 'load' ):
    '''
    FUNCTION:
    Train and Implement the above CNN models, output the "predictions.csv" file as the final result

    PARAMETERS:
    pos_file_path: file that contains all the positive training samples
    neg_file_path: file that contains all the negative training samples
    test_file_path: file that contains all the unlabeled test samples
    glove_method: choose to use which kind of glove embedding, value of this parameter can be: ["trained"; "pretrained; "merged"]
    parameter: choose to train the model from zero or to load the pretrained parameters. value of this parameter can be: ["train", "load"]

    return:
    NO RETURN
    '''
    #Get the training samples from files
    with open(pos_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_pos = [x.strip() for x in content] 

    with open(neg_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_neg = [x.strip() for x in content] 

    documents = np.concatenate((original_documents_pos, original_documents_neg), axis = 0)

    num_pos = len(original_documents_pos)
    num_neg = len(original_documents_neg)

    with open(test_file_path, encoding = 'utf8') as f:
        content_test = f.readlines()
            
    test_documents = [x.strip() for x in content_test] 

    #get the selected type of glove embedding
    print("loading glove embedding")
    vocabulary, word_embedding, vector_dict = glove_embedding.get_embedding(glove_method)
    print("loading compelete")

    tknzr = TweetTokenizer()

    #Set the length of padding and values of padding
    MAX_LEN = 50
    padding = np.zeros(len(word_embedding[0]))
    x = []
    y = []
    
    #get the vectors that represent text from words vector
    print("Building vectors for documents")
    for index, doc in enumerate(documents):
        vlist = [vector_dict[token] for token in tknzr.tokenize(doc) if token in vector_dict]
        while len(vlist) < MAX_LEN:
            vlist.append(padding)
        if len(vlist) == MAX_LEN:
            if index < num_pos:
                y.append(1)
            else:
                y.append(0)
            x.append(vlist)
    print("Building compelete")  
    
    #shuffle the training samples for chunks deviding
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    
    #Set hyperparameters for the training model
    CHUNK_SIZE = 100000
    BATCH_SIZE = 256
    EPOCHES = 50
    LEARNING_RATE = 0.0005

    total_num = int(len(x)/CHUNK_SIZE)+1
    cnn = CNN()
    #use GPU to accerlate the training process
    cnn = cnn.cuda()
    print(cnn)

    #If choose to train from zero
    if parameter == 'train':
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        for chunk_num in range(total_num):
            optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
            #set a decreasing learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
            #Since the size of the entire training set is large, we need to split it and train it chunk by chunk
            if chunk_num != (total_num-1):
                chunk_x = x[chunk_num*CHUNK_SIZE : (chunk_num+1)*CHUNK_SIZE] 
                chunk_y = y[chunk_num*CHUNK_SIZE : (chunk_num+1)*CHUNK_SIZE]
            else:
                chunk_x = x[chunk_num*CHUNK_SIZE:]
                chunk_y = y[chunk_num*CHUNK_SIZE:]
            
            chunk_x = torch.tensor(chunk_x, dtype=torch.float)
            chunk_x = chunk_x.unsqueeze(1)
            chunk_y = torch.tensor(chunk_y)

            dataset = Data.TensorDataset(chunk_x, chunk_y)

            train_size = int(len(chunk_x) * 0.98)
            test_size = len(chunk_x) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            test_x = test_dataset[:][0]
            test_y = test_dataset[:][1]
            
            train_loader= Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

            if chunk_num ==0 :
                current_epoches = EPOCHES
            else:
                current_epoches = int(EPOCHES/2)
            for epoch in range(current_epoches):
                for step, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                    output = cnn(batch_x)  # batch_x=[50,1,28,28]
                    loss = loss_function(output, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    # print the current result to montitor the training process
                    if step % 50 == 0:
                        test_x = test_x.cuda()
                        test_output = cnn(test_x)
                        test_output = test_output.cpu()
                        pred_y = torch.max(test_output, 1)[1].data.numpy()
                        accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                        loss = loss.cpu()
                        print('Chunk: ', chunk_num, '/', total_num, ' Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
                        print("Current learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])
        
        #save all the weights of the model into a file for future use
        torch.save(cnn.state_dict(), '../data/CNN/self_implemented_cnn.pth')
        print("train_finished and all parameters are saved")
    
    #choose to load a pretrained weights file
    elif parameter == 'load':
        cnn.load_state_dict(torch.load('../data/CNN/self_implemented_cnn.pth'))
    
    else:
        print("wrong cnn train method!")
    
    #Construct the text vectors of test data from word embediing
    test_documents_x = []
    for index, doc in enumerate(test_documents):
        vlist = [vector_dict[token] for token in tknzr.tokenize(doc) if token in vector_dict]
        while len(vlist) < MAX_LEN:
            vlist.append(padding)
        test_documents_x.append(vlist)   
    
    test_documents_x = torch.tensor(test_documents_x, dtype=torch.float)
    test_documents_x = test_documents_x.unsqueeze(1)

    res = []
    
    #predict labels for test data chunk by chunk
    print("start to predict labels for test data")
    for i in range(40):
        output_x = test_documents_x[i*250:(i+1)*250].cuda()
        output_y = cnn(output_x)
        output_y = output_y.cpu()
        pred_y = torch.max(output_y, 1)[1].data.numpy()
        res.append(pred_y)
    
    res = np.array(res)
    res = res.reshape(10000)
    res[res==0] = -1
    OUTPUT_PATH = '../output/submission.csv' # TODO: fill in desired name of output file for submission
    ids_test = np.arange(1, len(test_documents)+1, 1)
    helpers.create_csv_submission(ids_test, res, OUTPUT_PATH)
    print("submission File created!")
    



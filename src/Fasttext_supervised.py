import fasttext
import numpy as np
import random
import helpers


def transform_labels(with_suffix):
    #Convert the resulting labels from fasttext format with suffix to -1 and 1 values
    res = []
    for i, y in enumerate(with_suffix):
        if y[0] == '__label__1':
            res.append(1)
        else :
            res.append(-1)
    return res


def fasttext_supervised(pos_file_path, neg_file_path, test_file_path, dim = 200, epoch = 50, ngrams = 4):
    '''
    FUNCTION:
    Use the fastext's supervised method to perform classification task directly

    PARAMETERS:
    pos_file_path: file that contains all the positive training samples
    neg_file_path: file that contains all the negative training samples
    test_file_path: file that contains all the unlabeled test samples
    dim: the dimension of the word embedding vector
    epoch: the epoches of training for the fasttext supervised training
    ngrams: windowsize when constructing the word embedding

    return:
    NO RETURN
    '''
    #Fasttext's supervised method require txt files with specific format of label to train and test
    labeled_document = "./data/Fasttext/fasttext_supervised_labeled.txt"
    train_document = "./data/Fasttext/fasttext_supervised_train.txt"
    val_document = "./data/Fasttext/fasttext_supervised_val.txt"

    #Open files used to train the model
    with open(pos_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_pos = [x.strip() for x in content] 

    with open(neg_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_neg = [x.strip() for x in content] 

    documents = np.concatenate((original_documents_pos, original_documents_neg), axis = 0)

    num_pos = len(original_documents_pos)
    num_neg = len(original_documents_neg)

    #Reconstruct the data to  the foramat that is required by fasttext, and then store it to a file
    with open(labeled_document,"w", encoding='utf-8') as f:
        for i,str in enumerate(documents):
            if i < num_pos:
                f.write('__label__' + '1' + '\t' + str + '\n')
            else:
                f.write('__label__' + '-1' + '\t' + str + '\n')
    
    #shuffle all the texts and split them into tarin_set and validation_set
    with open(labeled_document, encoding='utf-8') as f:
        content = f.readlines()

    preprocessed_documents = [x.strip() for x in content] 
    random.shuffle (preprocessed_documents)

    with open(train_document,"w", encoding='utf-8') as f:
        for i,str in enumerate(preprocessed_documents[0:int(len(preprocessed_documents)*0.9)]):
            f.write(str + '\n')

    with open(val_document,"w", encoding='utf-8') as f:
        for i,str in enumerate(preprocessed_documents[int(len(preprocessed_documents)*0.9):len(preprocessed_documents)]):
            f.write(str + '\n')
    
    #train the supervised model
    classifier = fasttext.train_supervised(train_document, dim = dim, epoch =epoch, word_ngrams = ngrams)
    
    #Check the model locally
    result = classifier.test(val_document)
    print ('val num:', result[0])
    print ('acc:', result[1])
    print ('recall rate:', result[2])

    #Read the test file and predict the labels we need
    with open(test_file_path, encoding = 'utf8') as f:
        content_test = f.readlines()
            
    test_documents = [x.strip() for x in content_test] 
    labels = classifier.predict(test_documents)
    labels = list(labels)

    transform_label = transform_labels(labels[0])

    OUTPUT_PATH = '../output/submission.csv' # TODO: fill in desired name of output file for submission
    ids_test = np.arange(1, len(test_documents)+1, 1)
    helpers.create_csv_submission(ids_test, transform_label, OUTPUT_PATH)
    print("submission File created!")

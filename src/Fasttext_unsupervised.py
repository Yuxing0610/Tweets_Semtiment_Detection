import fasttext
import numpy as np
import string
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
import random
import helpers

def fasttext_unsupervised(pos_file_path, neg_file_path, test_file_path, embedding_method = 'skipgram', if_visualize = False, aggregated_method = "mean", classify_method = "RR"):
    '''
    FUNCTION:
    Train and Implement a classifier based on skipgram or cbow word embedding, both of these methods are implemented based on fasttext

    PARAMETERS:
    pos_file_path: file that contains all the positive training samples
    neg_file_path: file that contains all the negative training samples
    test_file_path: file that contains all the unlabeled test samples
    embedding_method: choose to use which kind of  embedding, value of this parameter can be: ["skipgram", "cbow"]
    if_visualize: choose to whether to visualize the generated embedding with pca method
    aggregated_method: choose to use which method to aggregate, the value of this parameters can be: ["max", "mean", "min"]
    classify_method: choose to use which classifier, the value of this parameter can be: ["RR", "LR", "SVM"]

    return:
    NO RETURN
    '''
    #because fasttext takes a txt file for traning, so we need to generate a file that contains all the training samples
    train_document = "./data/Fasttext/fastext_unsupervised_train.txt"

    #Get data used to train the model
    with open(pos_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_pos = [x.strip() for x in content] 

    with open(neg_file_path, encoding='utf8') as f:
        content = f.readlines()

    original_documents_neg = [x.strip() for x in content] 

    documents = np.concatenate((original_documents_pos, original_documents_neg), axis = 0)

    num_pos = len(original_documents_pos)
    num_neg = len(original_documents_neg)

    #Construct file for fastext unsupervised training
    with open(train_document, "w", encoding='utf-8') as f:
        for str in documents:
            f.write(str + "\n")
    
    #train the word embedding
    model = fasttext.train_unsupervised(train_document, model = embedding_method)
    vocabulary = model.words
    word_embeddings = np.array([model[word] for word in vocabulary])

    if if_visualize:
        #visualize the word embedding
        helpers.visualize_embedding("./word_embedding.jpg", vocabulary, word_embeddings)

    #construct the sentence embedding based on word embedding
    vector_dict = dict(zip(vocabulary, word_embeddings))
    aggregated_doc_vectors = helpers.word2text(documents, vector_dict, word_embeddings.shape[1], aggregated_method)

    #get label vector
    y_pos = np.ones(num_pos)
    y_neg = np.ones(num_neg)
    y_neg = y_neg-2
    y = np.concatenate((y_pos, y_neg),axis=0).reshape(len(documents), 1)
    data = np.concatenate((y, aggregated_doc_vectors), axis = 1)

    #shuffle the data and split them into train and validation set
    np.random.shuffle(data)
    data_train = data[ : int(0.9*len(data))]
    data_val = data[int(0.9*len(data)): ]
    train_y = data_train[:,0]
    train_x = data_train[:, 1:]
    val_y = data_val[:,0]
    val_x = data_val[:, 1:]

    #Use different classifier to perform the classification task
    if classify_method == 'LR':
        model = LogisticRegression(solver='sag', max_iter = 1000)
        model.fit(train_x, train_y)
    
    elif classify_method == 'RR':
        model = RidgeClassifier(alpha = 0.01)
        model.fit(train_x, train_y)

    # It takes forever to train SVM models on large dataset, so we select a portion of it to train
    elif classify_method == 'SVM':
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(train_x[:100000], train_y[:100000])
        val_x = val_x[:1000]
        val_y = val_y[:1000]

    else:
        print("No such classifier!")
        return

    #test the method locally
    predict_val = model.predict(val_x)
    accuracy = accuracy_score(val_y, predict_val)
    print('Accuracy is : ', accuracy)

    #prediction on test data
    with open(test_file_path, encoding = 'utf8') as f:
        content_test = f.readlines()
            
    test_documents = [x.strip() for x in content_test] 
    aggregated_doc_vectors_test = helpers.word2text(test_documents, vector_dict, word_embeddings.shape[1], aggregated_method)

    test_y = model.predict(aggregated_doc_vectors_test)
    OUTPUT_PATH = '../output/submission.csv' # TODO: fill in desired name of output file for submission
    ids_test = np.arange(1, len(test_documents)+1, 1)
    helpers.create_csv_submission(ids_test, test_y, OUTPUT_PATH)
    print("submission File created!")

    

    
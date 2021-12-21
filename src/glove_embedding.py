import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import helpers

def get_embedding(glove_method):
    '''
    FUNCTION:
    get the glove embedding

    PARAMETERS:
    glove_methond: choose to use which kind of glove embedding, value of this parameter can be: ["trained"; "pretrained; "merged"]

    RETURN:
    vocabularly: all the vocaularies in the dictionary
    word_embedding: all embedded values
    vector_dict: dictionary wih vocabulary as key and word_embedding as value
    '''
    #pretrained word embedding of GLOVE
    pretrained_embedding = "../data/Glove/glove.twitter.27B.200d.txt"
    #glove_embedding trained by ourselves with window size of 4
    trained_embedding = "../data/Glove/vectors_4.txt"
    
    #if choose to use the pretrained word embedding
    if glove_method == 'pretrained':
        with open(pretrained_embedding, encoding='utf8') as f:
            content = f.readlines()

        we_glove_pretrained = [x.strip() for x in content]
        vocabulary = []
        word_embedding = []
        for i, str in enumerate(we_glove_pretrained):
            temp = []
            vocabulary.append(str.split()[0])
            for value in str.split()[1:]:
                temp.append(float(value))
            word_embedding.append(temp)

        vector_dict = dict(zip(vocabulary, word_embedding))
    
    #if choose to use the self-trained word embedding
    elif glove_method == 'trained':
        with open(trained_embedding, encoding='utf8') as f:
            content = f.readlines()

        we_glove_trained = [x.strip() for x in content] 
        vocabulary = []
        word_embedding = []
        for i, str in enumerate(we_glove_trained):
            temp = []
            vocabulary.append(str.split()[0])
            for value in str.split()[1:]:
                temp.append(float(value))
            word_embedding.append(temp)

        vector_dict = dict(zip(vocabulary, word_embedding))
    
    #if choose to merge two kinds of embedding
    elif glove_method == 'merged':
        with open(pretrained_embedding, encoding='utf8') as f:
            content = f.readlines()

        we_glove_pretrained = [x.strip() for x in content]

        with open(trained_embedding, encoding='utf8') as f:
            content = f.readlines()

        we_glove_trained = [x.strip() for x in content] 

        vocabulary1 = []
        word_embedding1 = []
        for i, str in enumerate(we_glove_pretrained):
            temp = []
            vocabulary1.append(str.split()[0])
            for value in str.split()[1:]:
                temp.append(float(value))
            word_embedding1.append(temp)

        vector_dict1 = dict(zip(vocabulary1, word_embedding1))

        vocabulary2 = []
        word_embedding2 = []
        for i, str in enumerate(we_glove_trained):
            temp = []
            vocabulary2.append(str.split()[0])
            for value in str.split()[1:]:
                temp.append(float(value))
            word_embedding2.append(temp)

        print("Seperate construction done!")

        vector_dict2 = dict(zip(vocabulary2, word_embedding2))

        #Merge two embeddings together, update the trained value with pretrained value
        vocabulary = vocabulary1 + vocabulary2
        word_embedding = word_embedding1 + word_embedding2
        vector_dict = vector_dict1.copy()
        vector_dict.update(vector_dict2)

    else:
        print("Wrong glove method!")
        return
    
    return vocabulary, word_embedding, vector_dict



def glove_embedding(pos_file_path, neg_file_path, test_file_path, glove_method = 'merged', if_visualize = False, aggregated_method = "mean", classify_method = "RR"):
    '''
    FUNCTION:
    Train and Implement a classifier based on GLOVE word embedding

    PARAMETERS:
    pos_file_path: file that contains all the positive training samples
    neg_file_path: file that contains all the negative training samples
    test_file_path: file that contains all the unlabeled test samples
    glove_method: choose to use which kind of glove embedding, value of this parameter can be: ["trained"; "pretrained; "merged"]
    if_visualize: choose to whether to visualize the generated embedding with pca method
    aggregated_method: choose to use which method to aggregate, the value of this parameters can be: ["max", "mean", "min"]
    classify_method: choose to use which classifier, the value of this parameter can be: ["RR", "LR", "SVM"]

    return:
    NO RETURN
    '''
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

    print("loading glove embedding")
    vocabulary, word_embedding, vector_dict = get_embedding(glove_method)
    print("fininsh loading glove embedding")

    if if_visualize:
        #visualize the word embedding
        helpers.visualize_embedding("./word_embedding.jpg", vocabulary, word_embedding)

    print("computing vectors for texts")
    aggregated_doc_vectors = helpers.word2text(documents, vector_dict, len(word_embedding[0]), aggregated_method)
    print("finish computing vectors for texts")

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
        print("traing with linear regression")
        model = LogisticRegression(solver='sag', max_iter = 1000)
        model.fit(train_x, train_y)
    
    elif classify_method == 'RR':
        model = RidgeClassifier(alpha = 0.01)
        print("traing with Ridge regression")
        model.fit(train_x, train_y)

    # It takes forever to train SVM models on large dataset, so we select a portion of it to train
    elif classify_method == 'SVM':
        model = svm.SVC(C=1.0, kernel='linear')
        print("traing with SVM")
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
    aggregated_doc_vectors_test = helpers.word2text(test_documents, vector_dict, len(word_embedding[0]), aggregated_method)

    test_y = model.predict(aggregated_doc_vectors_test)
    OUTPUT_PATH = '../output/submission.csv' # TODO: fill in desired name of output file for submission
    ids_test = np.arange(1, len(test_documents)+1, 1)
    helpers.create_csv_submission(ids_test, test_y, OUTPUT_PATH)
    print("submission File created!")
    
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import numpy as np
from nltk.tokenize import TweetTokenizer, word_tokenize
    
def create_csv_submission(ids, y_pred, name):
    """
    FUNCTION:
    Creates an output file in .csv format for submission to Kaggle or AIcrowd

    PARAMETERS:
    ids (event ids associated with each prediction)
    y_pred (predicted class labels)
    name (string name of .csv output file to be created)

    RETURN:
    NO return
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def visualize_embedding(file_path, vocabulary, word_embeddings):
    '''
    FUNCTION:
    Take a set of embedded words, and use PCA to visualize the distribution of embedded values of differnet words in 2-D

    Parameters:
    file_path: where to save the resulted figure
    vocabulary: set of words
    word_embeddings: embedded values of the words mentioned above

    RETURN:
    NO return
    '''

    print("Start to visualize word embedding in 2-dimension:")
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, init = 'pca') 
    vis_data = tsne.fit_transform(word_embeddings)

    vis_data_x = vis_data[:,0]
    vis_data_y = vis_data[:,1]

    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(40, 40)) 
    plt.scatter(vis_data_x, vis_data_y)

    for label, x, y in zip(vocabulary, vis_data_x, vis_data_y):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.savefig(file_path)
    print("Figure saved!")
    

def aggregate_vector_list(vlist, aggfunc):
    '''
    FUNCTION:
    aggregate a set of word vectors to a single vector represents a sentence

    PARAMETERS:
    vlist: the list of embedded values
    aggfunv: choose to use which method to aggregate, the value of this parameters can be: ["max", "mean", "min"]

    RETURN:
    the vector that represents the sentence
    '''
    if aggfunc == 'max':
        return np.array(vlist).max(axis=0)
    elif aggfunc == 'min':
        return np.array(vlist).min(axis=0)
    elif aggfunc == 'mean':
        return np.array(vlist).mean(axis=0)
    else:
        return np.zeros(np.array(vlist).shape[1])


def word2text(documents, vector_dict, dim, method):
    '''
    FUNCTION:
    compute a vector representation for each sentence in the documents

    PARAMETERS:
    documents: all the documents to be computed
    vector_dict: a dictionary which keys are different words and values are vectors represent those words
    dim: dimension of the embedded vector
    method: choose to use which method to aggregate, the value of this parameters can be: ["max", "mean", "min"]

    RETURN:
    the vectors that represent all sentences in the given documents
    '''
    tknzr = TweetTokenizer()
    aggregated_doc_vectors = np.zeros((len(documents), dim))
    for index, doc in enumerate(documents):
        vlist = [vector_dict[token] for token in tknzr.tokenize(doc) if token in vector_dict]
        if(len(vlist) < 1):
            continue 
        else:
            aggregated_doc_vectors[index] = aggregate_vector_list(vlist, method)
    return aggregated_doc_vectors
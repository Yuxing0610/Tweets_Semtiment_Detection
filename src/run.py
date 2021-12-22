from absl import flags, app
from xlnet import xlnet
from bert import BERT
from Fasttext_supervised import fasttext_supervised
from Fasttext_unsupervised import fasttext_unsupervised
from glove_embedding import glove_embedding
from self_implemented_cnn import self_implemented_cnn

flags.DEFINE_string('pos_file_path', default='../data/twitter-datasets/train_pos.txt', help='pos_file_path')   
flags.DEFINE_string('neg_file_path', default='../data/twitter-datasets/train_neg.txt', help='neg_file_path')
flags.DEFINE_string('test_file_path', default='../data/twitter-datasets/test_data_processed.txt', help='test_file_path')

# bert, bertweet, transformer, xlnet
flags.DEFINE_string('model', default='bertweet', help='model')  #Use what kind of model
flags.DEFINE_bool('load_model', default=True, help='load_model')   #To load existed pth file or trian the model
flags.DEFINE_string('load_model_path', default='', help='load_model_path')  #Path of the pth file
flags.DEFINE_integer('num_epoch', default=0, help='num_epoch')  
flags.DEFINE_integer('batch_size', default=64, help='batch_size')
flags.DEFINE_integer('max_token_length', default=64, help='max_token_length')
flags.DEFINE_float('lr', default=2e-5, help='lr')

# embedding methods
flags.DEFINE_string('glove_method', default='merged', help='glove_method')  #method to get GloVe embedding, [merged, pretrained, trained]
flags.DEFINE_bool('glove_visualize', default=False, help='glove_visualize')  #if want to visualize the embedding result using PCA method
flags.DEFINE_string('aggregated_method', default='mean', help='aggregated_method')  #How to aggregate word vectors to sentence vector, [max, min, mean]
flags.DEFINE_string('classify_method', default='RR', help='classify_method')  #Use what kinds of classifier, [LR(logistic regression), RR(ridge regression), SVM]
flags.DEFINE_integer('fasttext_dim', default=200, help='fasttext_dim')  #the dimension of word vectors trained by fastext
flags.DEFINE_integer('fasttext_epoch', default=50, help='fasttext_epoch')  #epochs of fasttext training
flags.DEFINE_integer('ngrams', default=4, help='ngrams')  #value of N-grams
flags.DEFINE_string('embedding_method', default='skipgram', help='embedding_method')  #embedding method for fasttext, [skipgram, cbow]
flags.DEFINE_bool('fasttext_visualize', default=False, help='fasttext_visualize')  #if want to visualize the embedding result using PCA method
flags.DEFINE_string('cnn_parameter_option', default='load', help='cnn_parameter_option')  #To load existed pth file or trian the model, [load, train]


FLAGS = flags.FLAGS


def main(dummy_args):
    del dummy_args
    
    if FLAGS.model == 'bert' or FLAGS.model == 'bertweet':
        if FLAGS.load_model:
            if not FLAGS.load_model_path:
                if FLAGS.model == 'bertweet':
                    FLAGS.load_model_path = '../data/BERTweet/'
                else:
                    FLAGS.load_model_path = '../data/BERT/'
        BERT(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, FLAGS.load_model_path, batch_size=FLAGS.batch_size, epochs=FLAGS.num_epoch, 
             model_load_flag=FLAGS.load_model, model_type=FLAGS.model)
    
    elif FLAGS.model == 'xlnet':
        if FLAGS.load_model:
            if not FLAGS.load_model_path:
                FLAGS.load_model_path = '../data/XLNet/model.pth'
        xlnet(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, load_model=FLAGS.load_model, load_model_path=FLAGS.load_model_path, num_epoch=FLAGS.num_epoch, 
              batch_size=FLAGS.batch_size, max_token_length=FLAGS.max_token_length, lr=FLAGS.lr)
    
    elif FLAGS.model == 'glove_embedding':
        glove_embedding(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, glove_method=FLAGS.glove_method, if_visualize=FLAGS.glove_visualize, 
                        aggregated_method=FLAGS.aggregated_method, classify_method=FLAGS.classify_method)
        
    elif FLAGS.model == 'fasttext_supervised':
        fasttext_supervised(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, dim=FLAGS.fasttext_dim, epoch=FLAGS.fasttext_epoch, ngrams=FLAGS.ngrams)
        
    elif FLAGS.model == 'fasttext_unsupervised':
        fasttext_unsupervised(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, embedding_method=FLAGS.embedding_method, if_visualize=FLAGS.fasttext_visualize, 
                              aggregated_method = "mean", classify_method=FLAGS.classify_method)
    
    elif FLAGS.model == 'cnn':
        self_implemented_cnn(FLAGS.pos_file_path, FLAGS.neg_file_path, FLAGS.test_file_path, glove_method=FLAGS.glove_method, parameter=FLAGS.cnn_parameter_option)
    


if __name__ == "__main__":
    app.run(main)
    

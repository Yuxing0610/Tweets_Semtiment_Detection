import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
# from krovetzstemmer import Stemmer
import numpy as np
from absl import flags, app

flags.DEFINE_string('pos_input_file_path', default='../data/twitter-datasets/train_pos.txt', help='pos_input_file_path')
flags.DEFINE_string('neg_input_file_path', default='../data/twitter-datasets/train_neg.txt', help='neg_input_file_path')
flags.DEFINE_string('test_input_file_path', default='../data/twitter-datasets/test_data.txt', help='test_input_file_path')

flags.DEFINE_string('pos_output_file_path', default='../data/twitter-datasets/train_pos_processed.txt', help='pos_output_file_path')
flags.DEFINE_string('neg_output_file_path', default='../data/twitter-datasets/train_neg_processed.txt', help='neg_output_file_path')
flags.DEFINE_string('test_output_file_path', default='../data/twitter-datasets/test_data_processed.txt', help='test_output_file_path')

flags.DEFINE_bool('remove_duplicates_and_na', default=True, help='remove_duplicates_and_na')
flags.DEFINE_bool('process_emoticons', default=True, help='process_emoticons')
flags.DEFINE_bool('remove_stop_words', default=False, help='remove_stop_words')
flags.DEFINE_bool('process_numbers', default=True, help='process_numbers')
flags.DEFINE_bool('process_letter_repetition', default=True, help='process_letter_repetition')
flags.DEFINE_bool('process_abbreviation', default=True, help='process_abbreviation')
flags.DEFINE_bool('process_punctuations', default=False, help='process_punctuations')
flags.DEFINE_bool('word_lemmatization', default=True, help='word_lemmatization')
flags.DEFINE_bool('remove_conflicts', default=True, help='remove_conflicts')

FLAGS = flags.FLAGS

def main(dummy_args):
    del dummy_args
    
    pos_input_file_path = FLAGS.pos_input_file_path
    neg_input_file_path = FLAGS.neg_input_file_path
    test_input_file_path = FLAGS.test_input_file_path

    pos_output_file_path = FLAGS.pos_output_file_path
    neg_output_file_path = FLAGS.neg_output_file_path
    test_output_file_path = FLAGS.test_output_file_path

    remove_duplicates_and_na = FLAGS.remove_duplicates_and_na
    process_emoticons = FLAGS.process_emoticons
    remove_stop_words = FLAGS.remove_stop_words
    process_numbers = FLAGS.process_numbers
    process_letter_repetition = FLAGS.process_letter_repetition
    process_abbreviation = FLAGS.process_abbreviation
    process_punctuations = FLAGS.process_punctuations
    word_lemmatization = FLAGS.word_lemmatization
    remove_conflicts = FLAGS.remove_conflicts

    # Read data
    pos_train_data = []
    with open(pos_input_file_path, encoding='UTF-8') as f:
        for line in f:
            pos_train_data.append(line.strip())
    pos_df = pd.DataFrame(pos_train_data)
    pos_df = pos_df.rename(columns={0:'tweet'})
    pos_df = pos_df.dropna()

    neg_train_data = []
    with open(neg_input_file_path, encoding='UTF-8') as f:
        for line in f:
            neg_train_data.append(line.strip())
    neg_df = pd.DataFrame(neg_train_data)
    neg_df = neg_df.rename(columns={0:'tweet'})
    neg_df = neg_df.dropna()

    test_data = []
    with open(test_input_file_path, encoding='UTF-8') as f:
        for line in f:
            test_data.append(','.join(line.strip().split(',')[1:]))
    test_df = pd.DataFrame(test_data)
    test_df = test_df.rename(columns={0:'tweet'})

    # Dop duplicates
    if remove_duplicates_and_na:
        pos_df = pos_df.drop_duplicates(keep='first').reset_index(drop=True)
        neg_df = neg_df.drop_duplicates(keep='first').reset_index(drop=True)

    # Process emoticons
    if process_emoticons:
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
            ])

        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
            ])

        def process_emoticons(tweet):
            tokens = []
            for token in tweet.split():
                if token in emoticons_happy:
                    tokens.append('<emoticonhappy>')
                elif token in emoticons_sad:
                    tokens.append('<emoticonsad>')
                else:
                    tokens.append(token)
            return (' '.join(tokens)).strip()

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: process_emoticons(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: process_emoticons(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: process_emoticons(tweet))

    # Remove stop words
    if remove_stop_words:
    #     stop_words = set([
    #         'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    #         'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
    #         'were', 'will', 'with', '<user>', '<url>'
    #         ])
        stop_words = set(['a', 'an', '<url>'])

        def remove_stop_words(tweet):
            tokens = []
            for token in tweet.split():
                if token in stop_words:
                    pass
                else:
                    tokens.append(token)
            return ' '.join(tokens)

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: remove_stop_words(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: remove_stop_words(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: remove_stop_words(tweet))

    # Process numbers
    if process_numbers:
        def process_numbers(tweet):
            tweet = re.sub('[0-9]{5,}', '<numhuge>', tweet)
            tweet = re.sub('[0-9]{4}', '<numlarge>', tweet)
            tweet = re.sub('[0-9]{3}', '<nummedium>', tweet)
            tweet = re.sub('[0-9]{2}', '<numsmall>', tweet)
            return tweet

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: process_numbers(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: process_numbers(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: process_numbers(tweet))

    # Process repetition of letter
    if process_letter_repetition:
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"

        def process_repetition(tweet):
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
            return tweet

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: process_repetition(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: process_repetition(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: process_repetition(tweet))

    # Process abbreviation
    if process_abbreviation:
        def process_abbreviation(tweet):
            tweet = re.sub('n\'t', ' not', tweet)
            tweet = re.sub('i\'m', 'i am', tweet)
            tweet = re.sub('\'re', ' are', tweet)
            tweet = re.sub('it\'s', 'it is', tweet)
            tweet = re.sub('that\'s', 'that is', tweet)
            tweet = re.sub('\'ll', ' will', tweet)
            tweet = re.sub('\'l', ' will', tweet)
            tweet = re.sub('\'ve', ' have', tweet)
            tweet = re.sub('\'d', ' would', tweet)
            tweet = re.sub('he\'s', 'he is', tweet)
            tweet = re.sub('what\'s', 'what is', tweet)
            tweet = re.sub('who\'s', 'who is', tweet)
            tweet = re.sub('\'s', '', tweet)
            return tweet

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: process_abbreviation(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: process_abbreviation(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: process_abbreviation(tweet))

    # Process punctuation
    if process_punctuations:
        punctuations = set([
            '\"', '#', '$', '%', '&', '\\', '\'', '(', ')', '*', '+',
            ',', '-', '.', '/', ':', ';', '<', '=', '>', '@', '[',
            ']', '^', '_', '`', '{', '|', '}', '~'
            ])

        def process_punctuations(tweet):
            tokens = []
            for token in tweet.split():
                if token in punctuations:
                    pass
                else:
                    tokens.append(token)
            return (' '.join(tokens)).strip()

        pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: process_punctuations(tweet))
        neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: process_punctuations(tweet))
        test_df['tweet'] = test_df['tweet'].apply(lambda tweet: process_punctuations(tweet))

    # Word stemming and lemmatization
    if word_lemmatization:
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        # stemmer require pre-installed local library
        # stemmer = Stemmer()

        use_lemmatizer = True
        # use_stemmer = True

        def word_lemmatization(tweet):
            tokens = []
            for token in tweet.split():
                tokens.append(lemmatizer.lemmatize(token))
            return ' '.join(tokens)

        # def word_stemming(tweet):
        #     tokens = []
        #     for token in tweet.split():
        #         tokens.append(stemmer.stem(token))
        #     return ' '.join(tokens)

        if use_lemmatizer:
            pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: word_lemmatization(tweet))
            neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: word_lemmatization(tweet))
            test_df['tweet'] = test_df['tweet'].apply(lambda tweet: word_lemmatization(tweet))

        # if use_stemmer:
        #     pos_df['tweet'] = pos_df['tweet'].apply(lambda tweet: word_stemming(tweet))
        #     neg_df['tweet'] = neg_df['tweet'].apply(lambda tweet: word_stemming(tweet))
        #     test_df['tweet'] = test_df['tweet'].apply(lambda tweet: word_stemming(tweet))

    # After all previous processing
    if remove_duplicates_and_na:
        pos_df = pos_df.dropna()
        pos_df = pos_df.drop_duplicates(keep='first').reset_index(drop=True)
        neg_df = neg_df.dropna()
        neg_df = neg_df.drop_duplicates(keep='first').reset_index(drop=True)

    # Remove conflicts
    if remove_conflicts:
        merged_df = pd.concat([pos_df, neg_df])
        conflict_df = merged_df[merged_df.duplicated(['tweet'], keep=False)]

        pos_df = pd.concat([pos_df, conflict_df])
        pos_df = pos_df.drop_duplicates(keep=False).reset_index(drop=True)
        neg_df = pd.concat([neg_df, conflict_df])
        neg_df = neg_df.drop_duplicates(keep=False).reset_index(drop=True)

    np.savetxt(pos_output_file_path, pos_df.values, fmt='%s')
    np.savetxt(neg_output_file_path, neg_df.values, fmt='%s')
    np.savetxt(test_output_file_path, test_df.values, fmt='%s')

    print('Size of pos data after pre-processing: ', pos_df.size)
    print('Size of neg data after pre-processing: ', neg_df.size)
    print('Size of test data after pre-processing: ', test_df.size)

if __name__ == "__main__":
    app.run(main)

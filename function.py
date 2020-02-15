import string

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from nltk import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    return html_free


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def remove_punctuation(text):
    # "".join will join the list of letters back together as words where there
    # are no spaces.
    # does not account user typos
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()


def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text


# Instantiate Stemmer
stemmer = PorterStemmer()


def word_stemmer(text):
    # stem_text = " ".join([stemmer.stem(i) for i in text])
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text


def call_all():
    # read in csv file, create DataFrame and check shape
    df = pd.read_csv('train_small.csv')
    print(df.shape)
    df['comment_text'] = df['comment_text'].apply(lambda x: remove_html(x))

    df['comment_text'] = df['comment_text'].apply(lambda x: remove_urls(x))

    df['comment_text'] = df['comment_text'].apply(lambda x: remove_numbers(x))

    df['comment_text'] = df['comment_text'].apply(
        lambda x: remove_punctuation(x))

    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

    df['comment_text'] = df['comment_text'].apply(
        lambda x: tokenizer.tokenize(x.lower()))

    df['comment_text'] = df['comment_text'].apply(lambda x: remove_stopwords(x))

    # df['comment_text'] = df['comment_text'].apply(lambda x: word_lemmatizer(x))

    df['comment_text'] = df['comment_text'].apply(lambda x: word_stemmer(x))

    print(df['comment_text'])

    return df['comment_text']


def create_models(corp):
    norm_corpus = np.array(corp)
    print(norm_corpus)
    # Bag of Words Model
    # make every word into numeric vectors, called the bag of words model.
    # Display it as a table
    cv = CountVectorizer(min_df=0., max_df=1.)
    cv_matrix = cv.fit_transform(norm_corpus)
    cv_matrix = cv_matrix.toarray()
    print(cv_matrix)

    # get all unique words in the corpus
    vocab = cv.get_feature_names()
    print(vocab)
    # show document feature vectors
    print(pd.DataFrame(cv_matrix, columns=vocab))

    # Bag of N-Grams Model
    # create a bigram model to see the frequency of phrases
    bv = CountVectorizer(ngram_range=(2, 2))
    bv_matrix = bv.fit_transform(norm_corpus)

    bv_matrix = bv_matrix.toarray()
    vocab = bv.get_feature_names()
    print(vocab)
    print(pd.DataFrame(bv_matrix, columns=vocab))

    # TF-IDF(Term Frequency-Inverse Document Frequency) Model
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(norm_corpus)
    tv_matrix = tv_matrix.toarray()

    vocab = tv.get_feature_names()
    print(vocab)
    print(pd.DataFrame(np.round(tv_matrix, 2), columns=vocab))
    return len(vocab)


if __name__ == '__main__':
    corpus = []
    list_sentences_train = call_all()

    for i in list_sentences_train:
        corpus.append(i)
    len_vocab = create_models(corpus)

    max_features = len_vocab
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    print(list_tokenized_train[:1])

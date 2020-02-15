import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Embedding, Input, LSTM
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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

    # print(df['comment_text'])

    return df['comment_text']


def create_models(corp):
    norm_corpus = np.array(corp)
    # print(norm_corpus)
    # Bag of Words Model
    # make every word into numeric vectors, called the bag of words model.
    # Display it as a table
    cv = CountVectorizer(min_df=0., max_df=1.)
    cv_matrix = cv.fit_transform(norm_corpus)
    cv_matrix = cv_matrix.toarray()
    # print(cv_matrix)

    # get all unique words in the corpus
    vocab = cv.get_feature_names()
    # print(vocab)
    # show document feature vectors
    # print(pd.DataFrame(cv_matrix, columns=vocab))

    # Bag of N-Grams Model
    # create a bigram model to see the frequency of phrases
    bv = CountVectorizer(ngram_range=(2, 2))
    bv_matrix = bv.fit_transform(norm_corpus)

    bv_matrix = bv_matrix.toarray()
    vocab = bv.get_feature_names()
    # print(vocab)
    # print(pd.DataFrame(bv_matrix, columns=vocab))

    # TF-IDF(Term Frequency-Inverse Document Frequency) Model
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(norm_corpus)
    tv_matrix = tv_matrix.toarray()

    vocab = tv.get_feature_names()
    # print(vocab)
    # print(pd.DataFrame(np.round(tv_matrix, 2), columns=vocab))
    return len(vocab)


if __name__ == '__main__':
    df = pd.read_csv('train_small.csv')
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                    "identity_hate"]
    y = df[list_classes].values

    corpus = []
    list_sentences_train = call_all()

    for i in list_sentences_train:
        corpus.append(i)
    len_vocab = create_models(corpus)
    print(len_vocab)  # 1503

    max_features = 5000  # len_vocab
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    # print(list_tokenized_train[:1])

    totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
    # plt.hist(totalNumWords, bins=np.arange(0, 410, 10))
    # [0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
    # the distribution of the number of words in sentences.
    # print(plt.show())

    maxlen = 250
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

    # the inputs into our networks are our list of encoded sentences
    # By indicating an empty space after comma, we are telling Keras to infer
    # the number automatically.
    inp = Input(shape=(maxlen, ))

    embed_size = 128
    # The output of the Embedding layer is just a list of the coordinates of
    # the words in this vector space.
    x = Embedding(max_features, embed_size)(inp)
    # The embedding layer outputs a 3-D tensor of (None, 200, 128). Which is an
    # array of sentence(None means that it's size is inferred), and for each
    # words(200), there is an array of 128 coordinates in the vector space of
    # embedding.

    # What it does is going through the samples, recursively run the LSTM model
    # for 200 times, passing in the coordinates of the words each time.
    x = LSTM(60, return_sequences=True, name='lstm_layer')(x)

    # We reshape carefully to avoid throwing away data that is important to us,
    # and ideally we want the resulting data to be a good representative of the
    # original data.
    # We go through each patch of data, and we take the maximum values of each
    # patch
    x = GlobalMaxPool1D()(x)

    # With a 2D Tensor in our hands, we pass it to a Dropout layer which
    # indiscriminately "disable" some nodes so that the nodes in the next layer
    # is forced to handle the representation of the missing data and the whole
    # network could result in better generalization.
    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    # We are almost done! All is left is to define the inputs, outputs and
    # configure the learning process. We have set our model to optimize our
    # loss function using Adam optimizer, define the loss function to be
    # "binary_crossentropy" since we are tackling a binary classification.
    # In case you are looking for the learning rate, the default is set at
    # 0.001.
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # put our model to the test
    batch_size = 32
    epochs = 2
    model.fit(X_t, y, batch_size=batch_size, epochs=epochs,
              validation_split=0.1)

    print(model.summary())
    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_3rd_layer_output([X_t[:1]])[0]
    print(layer_output.shape)
    print(layer_output)
    # print layer_output to see the actual data

    # ========= LSTM Modeling =========

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['comment_text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['comment_text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    Y = pd.get_dummies(df[list_classes]).values
    print('Shape of label tensor:', Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,
                                                        random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    epochs = 2
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],
                                                                  accr[1]))

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    print(plt.show())

    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    print(plt.show())

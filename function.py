import pandas as pd
import nltk
nltk.download('wordnet')
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    html_free = soup.get_text()
    return html_free


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
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text


def call_all():
    # read in csv file, create DataFrame and check shape
    df = pd.read_csv('train_small.csv')
    print(df.shape)

    # remove punctuation
    df['comment_text'] = df['comment_text'].apply(lambda x: remove_punctuation(x))

    # Instantiate Tokenizer
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

    df['comment_text'] = df['comment_text'].apply(lambda x: tokenizer.tokenize(x.lower()))
    # print(df['comment_text'].head(20))

    # remove stop words
    df['comment_text'] = df['comment_text'].apply(lambda x: remove_stopwords(x))
    # print(df['comment_text'].head(10))

    # compare stemming and lemmatizer to see which one works better before
    # assigning
    # print(df['comment_text'].apply(lambda x: word_lemmatizer(x)))
    # print(df['comment_text'].apply(lambda x: word_stemmer(x)))

    df['comment_text'] = df['comment_text'].apply(lambda x: word_lemmatizer(x))

    print(df['comment_text'])


if __name__ == '__main__':
    call_all()

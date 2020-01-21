
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('train.csv')
def clean_data(text):
    # Clean the images
    text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

    # Clean all the punctuation
    text = "".join([c for c in text if c not in string.punctuation])

    # Convert into lowercase
    text = text.lower()

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer.tokenize(text)


    #text = "".join([w for w in text if w not in stopwords.words('english')])
    return text


if __name__ == '__main__':
    df["comment_text"] = [clean_data(text) for text in df['comment_text']]
    print(df["comment_text"].head())


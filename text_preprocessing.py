import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd

alay_df = pd.read_csv('data/new_kamusalay.csv', 
                      encoding = 'latin-1', 
                      header = None)

alay_df.rename(columns={0: 'original', 
                        1: 'replacement'},
               inplace = True)

alay_dict_map = dict(zip(alay_df['original'], alay_df['replacement']))

def normalize_alay(text):
  return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])
  
def convert_lower_case(text):
  return text.lower()

def remove_stop_words(text):
  stop_words = stopwords.words('indonesian')
  words = word_tokenize(str(text))
  new_text = ""
  for w in words:
    if w not in stop_words and len(w) > 1:
      new_text = new_text + " " + w
  return new_text

def remove_unnecessary_char(text):
  text = re.sub('\n',' ',text) # Remove every '\n'
  text = re.sub('rt',' ',text) # Remove every retweet symbol
  text = re.sub('user',' ',text) # Remove every username
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
  text = re.sub('  +', ' ', text) # Remove extra spaces
  
  text = re.sub('x9f', ' ', text)
  text = re.sub('x98', ' ', text)
  text = re.sub('xf0', ' ', text)

  text = re.sub(' ya ', ' ', text)
  text = re.sub('x82', ' ', text)
  text = re.sub('uniform', ' ', text)
  text = re.sub('resource', ' ', text)

  text = re.sub('xe2', ' ', text)
  text = re.sub('x80', ' ', text)
  text = re.sub('x91', ' ', text)
  text = re.sub('x8c', ' ', text)

  text = re.sub('locator', ' ', text)
  return text

def remove_punctuation(text):
  symbols = string.punctuation
  for i in range(len(symbols)):
    text = text.replace(symbols[i], ' ')
    text = text.replace("  ", " ")
  text = text.replace(',', '')
  return text

def stemming(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  
  tokens = word_tokenize(str(text))
  new_text = ""
  for w in tokens:
    new_text = new_text + " " + stemmer.stem(w)
  return new_text

def preprocess(text):
  text = convert_lower_case(text)
  text = remove_unnecessary_char(text)
  text = remove_punctuation(text)
  text = remove_stop_words(text)
  text = normalize_alay(text)
  text = stemming(text)
  text = remove_stop_words(text)
  text = remove_punctuation(text)
  return text

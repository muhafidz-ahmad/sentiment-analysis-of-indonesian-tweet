import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

import text_preprocessing

def load_model():
  model = tf.keras.models.load_model('model.h5')
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  return [model, tokenizer]
  
def predict(model, text):
    # model[0] = model machine learning
    # model[1] = tokenizer
    labels_list=['Very Negative', 'Negative Hate Speech', 'Negative Abusive', 'Positive']
    dict_classes = dict(zip(range(len(labels_list)),
                            labels_list))
    
    # text preprocessing, include tokenizing
    preprocess_text = text_preprocessing.preprocess(text, stem=True)  # text_preprocessing
    text = model[1].texts_to_sequences([preprocess_text])   # tokenizer
    text = pad_sequences(text, padding='post',
                         maxlen=100, truncating='post')
    
    # predict the text
    prediction = model[0].predict(text, verbose=0)
    classes = np.argmax(prediction, axis = 1)
    pred_class = dict_classes[classes[0]]
    
    # create a table of predicted categories
    df = pd.Series(prediction[0].round(decimals=5) * 100, 
                   index=dict_classes.values()).sort_values(ascending=False)
    df = df.to_frame().reset_index()
    df = df.rename(columns={'index': 'sentiment',
                            0: 'probability'})
    
    return df, preprocess_text

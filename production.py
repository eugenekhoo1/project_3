#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
import pickle
import string
import streamlit as st
import re

# Define clean data function

wnl = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english')
punc_str = string.punctuation + "’"
def clean_data_lemm(text):
    # remove punctuation
    text = str(text)
    text_nopunc = ''.join(char.lower() for char in text if char not in punc_str)
    
    # tokenize
    tokenized = re.split('\W+', text_nopunc)
    
    # remove words with numbers
    text_remove_num = [re.sub(r"^\d+|\d$","",word) for word in tokenized]
    
    # lemmatize + remove stopwords
    text_clean = [wnl.lemmatize(word) for word in text_remove_num if word not in stopword]
    
    return text_clean

# Load TF-IDF vectorizer and Logistic Regression model

tfidf_vec, tfidf_lr = pickle.load(open('nlp.pickle', 'rb'))

# Define class

class classify():
    
    def __init__(self, text):
        self.text = text
        self.text_vec = tfidf_vec.transform(pd.Series(self.text))
        self.text_clean = clean_data_lemm(self.text)
        self.score = tfidf_lr.coef_
        self.dict_tfidf = tfidf_vec.vocabulary_
        print(f"{self.text} -- initialized")
        
    def predict_(self):
        if tfidf_lr.predict(self.text_vec)[0] == 0:
            return f"Depression"
        else:
            return f"Anxiety"
        
    def feature_(self):
        feature_dict = {
            'token' : [],
            'score' : []
        }
        for i in self.text_clean:
            try:
                """
                Try to add score in first, if not found (i.e. word is not in set of 10k words),
                move to the next word..
                """
                feature_dict['score'].append(self.score[0][self.dict_tfidf[i]]) 
                feature_dict['token'].append(i)
            except:
                pass
        df = pd.DataFrame(feature_dict).sort_values(by='score', key=abs, ascending=False)
        df.drop_duplicates(inplace=True)
        df.set_index('token', inplace=True)
        return df.head(10)
    
    def result_(self):
        print(self.predict_())
        print(self.feature_())


# Streamlit Implementation

st.set_page_config(page_title='Chippy')

# Header

with st.container():
    st.title('Chippy')
    st.subheader('Anxiety vs Depression')

with st.container():
    subreddit_text = st.text_input('Input Text', value='')

subreddit_obj = classify(subreddit_text)

def Predict():
    with st.container():
        st.markdown(f"Prediction: __{subreddit_obj.predict_()}__")
        st.write('Token Scores:', subreddit_obj.feature_())

def Clear():
    subreddit_text = ''

st.button('Predict', on_click=Predict)
st.button('Clear', on_click=Clear)


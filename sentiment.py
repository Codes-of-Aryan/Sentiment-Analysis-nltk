#Sentiment Analysis using nltk 
#Author: Aryan Agrawal 
#Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import os
import warnings

def strip_html(text): 
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def text_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def delete_special_characters(text): 
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def cleanData(text): 
    text = strip_html(text) 
    text = text_between_square_brackets(text)
    text = delete_special_characters(text)
    return text 

def porterStemmer(text):
    ps=nltk.porter.PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])

def remove_stopwords(text, is_lower_case=False):
    tokenizer=ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english') #English stopwords
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def eda(data): 
    print('Shape of the data is: ', data.shape)
    print('Column Names are: ', data.columns.values)
    print (data.describe) 
    print(data['sentiment'].value_counts()) 

def preprocess(data): 
    
    #Strip HTML, remove text between [] brackets and remove special characters. 
    data['review'] = data['review'].apply(cleanData)
    
    #Porter stemming
    data['review'] = data['review'].apply(porterStemmer)
    
    #Remove StopWords
    data['review']= data['review'].apply(remove_stopwords)
    
    return data


def main(): 
    train_data = pd.read_csv('.\IMDB Dataset.csv')
    eda(train_data) 
    train_data = preprocess(train_data) 

    norm_train_reviews = train_data['review'][:40000]
    norm_test_reviews = train_data['review'][40000:]

    tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    #transformed train reviews
    tv_train_reviews=tv.fit_transform(norm_train_reviews)
    #transformed test reviews
    tv_test_reviews=tv.transform(norm_test_reviews)

    lb=LabelBinarizer()
    #transformed sentiment data
    sentiment_data=lb.fit_transform(train_data['sentiment'])
    print(sentiment_data.shape)
    #Spliting the sentiment data
    train_sentiments=sentiment_data[:40000]
    test_sentiments=sentiment_data[40000:]
    
    mnb=MultinomialNB()

    #fitting the svm for tfidf features
    mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)
    print(mnb_tfidf)

    #Predicting the model for tfidf features
    mnb_tfidf_predict=mnb.predict(tv_test_reviews)
    print(mnb_tfidf_predict)

    #Classification report for tfidf features
    mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
    print(mnb_tfidf_report)


if __name__ == '__main__': 
    main() 
import seaborn as sns
import numpy as np
import pandas as pd
import re
from collections import Counter
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import itertools
nltk.download('punkt')
nltk.download('stopwords')

analyzer = SentimentIntensityAnalyzer()
#Function for generating the polarity of a single document 
def vader_sent_func(doc):
    sent_list = sent_tokenize(doc)
    vader_doc = []
    sent_index = []
    for i in range(len(sent_list)):
        vs_sent = analyzer.polarity_scores(sent_list[i])
        vader_doc.append(vs_sent)
        sent_index.append(i)
        
    # Get the output as a DataFrame    
    doc_df = pd.DataFrame(vader_doc)
    doc_df.insert(0, 'sent_index', sent_index)  
    doc_df.insert(doc_df.shape[1], 'sentence', sent_list)
    return(doc_df) 


# This function will yield the modified data as an output when called.
#Sentiment Analysis for the whole corpus
def vader_corpus_func(doc):    
    # Initializing the dataframe to add the output of single document
    vader_doc_df = pd.DataFrame(columns=['doc_index', 'sent_index', 'neg', 'neu', 'pos', 'compound', 'sentence'])     
    for i in range(len(doc)):
        vader_doc = vader_sent_func(doc.loc[i])
        vader_doc.insert(0,'doc_index', i)
        vader_doc_df = pd.concat([vader_doc_df, vader_doc], axis=0)
        
    return(vader_doc_df) 


#Generic function for generating any no. of grams based on the input n
def token_ngrams(sent, n):
    n_grams = ngrams(sent, n)
    return [ ' '.join(grams) for grams in n_grams]
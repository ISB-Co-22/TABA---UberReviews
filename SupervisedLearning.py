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

## Defining a general function for generating DTMs by top n tokens

def series2dtm(series0, min_df1=5, ngram_range1=(1,2), top_n=200):

    # Generate DTM matrix
    tf_vect = CountVectorizer(lowercase=False, min_df=min_df1, ngram_range=ngram_range1)
    dtm_tf = tf_vect.fit_transform(series0)
 
    # Reduce the DTM matrix to top n terms
    pd0 = pd.Series(dtm_tf.sum(axis=0).tolist()[0])
    ind0 = pd0.sort_values(ascending=False).index.tolist()[:top_n]
    feat0 = pd.Series(tf_vect.get_feature_names()).iloc[ind0]
    dtm_tf1 = dtm_tf[:,ind0].todense()
    dtm_df = pd.DataFrame(data=dtm_tf1, columns=feat0.tolist())

    # Generate TF-IDF matrix
    idf_vect = TfidfVectorizer(lowercase=False, min_df=min_df1, ngram_range=ngram_range1)
    dtm_idf = idf_vect.fit_transform(series0)

    # Reduce the TF-IDF matrix to top n terms
    pd0 = pd.Series(dtm_idf.sum(axis=0).tolist()[0])
    ind0 = pd0.sort_values(ascending=False).index.tolist()[:top_n]
    feat0 = pd.Series(idf_vect.get_feature_names()).iloc[ind0]
    dtm_idf1 = dtm_idf[:,ind0].todense()
    dtm_idf = pd.DataFrame(data=dtm_idf1, columns=feat0.tolist())

    return(dtm_df, dtm_idf)



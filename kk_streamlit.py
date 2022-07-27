import streamlit as st 
import time
import pandas as pd
import numpy as np
import seaborn as sns
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
import itertools
nltk.download('punkt')
nltk.download('stopwords')


from LoadData import *
from DataPreProcessing import *
from SentimentAnalysis import *
from SupervisedLearning import *

st.set_option('deprecation.showPyplotGlobalUse', False)

FILE_ADDRESS = st.sidebar.file_uploader('Upload file')
stopwords_eng = stopwords.words("English")

# Default landing page if no file has been uploaded.
if FILE_ADDRESS is None:
	st.title('Text Analytics Group Assignment Streamlit application.')
	st.write('Group Member:')
	st.write('Abhishek Chintamani (12120082)')
	st.write('Amit Shukla (12120102)')
	st.write('Rupali Agarwal (12120100)')
	st.write('Smaranika Sikdar (12120092)')
	st.write('Varun Ananthula (12120066)')
	
	st.header('To Start with, upload (csv) file using the sidebar on the left.')

    # If file has been uploaded
else:
	start_time = time.time()
	dataset = load_data(FILE_ADDRESS)  # Calling From LoadData.py
	Uber_reviews = dataset


	option = st.sidebar.selectbox('Navigation',
                                  ["Sentiment Analysis",  
                                   "N-Gram Analysis", "Supervised Learning"])

#Calling Functions from DataPreProcessing
	dh=MissingValues(Uber_reviews) # To check any missing values in the data , Calling From DataPreProcessing.py    
	Uber_reviews = DataCleanUp(Uber_reviews) #CleanUp Data    , Calling From DataPreProcessing.py   
	Uber_reviews = Uber_reviews.applymap(lambda s:s.lower() if type(s) == str else s) # Convert to Lower Case
    
  
          
 #Doing sentiment Analysis
	if option == "Sentiment Analysis":
		start_time = time.time()
		tab1, tab2,tab3, tab4, tab5,tab6 = st.tabs(["Uber Review", "Review Score","Review Score Describe", "Plotting Score", "WordCloud Positive", "WordCloud Negative"])
		SentAn_Uber_Review = Uber_reviews['Review']
		# Test run on the first review in Uber reviews data
		doc_df = vader_sent_func(SentAn_Uber_Review.loc[0])    
		Scores_Uber_Review = vader_corpus_func(SentAn_Uber_Review)
        
		Scores_Uber_Review.loc[Scores_Uber_Review['compound'] > 0.5, 'Polarity'] = 'Positive'
		Scores_Uber_Review.loc[Scores_Uber_Review['compound'] < -0.5, 'Polarity'] = 'Negative'
		Scores_Uber_Review.loc[(Scores_Uber_Review['compound'] > -0.5) & (Scores_Uber_Review['compound'] < 0.5), 'Polarity'] = 'Neutral'
		n_pos_sent = len(Scores_Uber_Review[Scores_Uber_Review['compound'] > 0.5])
		n_neg_sent = len(Scores_Uber_Review[Scores_Uber_Review['compound'] < -0.5])
		n_neu_sent = len(Scores_Uber_Review[(Scores_Uber_Review['compound'] > -0.5) & (Scores_Uber_Review['compound'] < 0.5)])
		n_sent = {'Positive':n_pos_sent, 'Negative':n_neg_sent, 'Neutral':n_neu_sent}

		plt.figure(figsize = (10, 7))
		sentiment = list(n_sent.keys())
		sent_values = list(n_sent.values())

		plt.bar(sentiment,sent_values,color ='blue',width = 0.4)
        
		st.write("Time taken to do Sentiment Analysis data in seconds:")
		st.write(time.time() - start_time)

		with tab1: 
			tab1.header("Uber Reviews")         
			tab1.write(SentAn_Uber_Review)

		with tab2:
			tab2.header("Uber Reviews Score")
			tab2.write(Scores_Uber_Review)
            
		with tab3: #Remove
			tab3.header("Uber Reviews Score Describe")
			tab3.write(Scores_Uber_Review.describe())

		with tab4:
			tab4.header("Plotting Figure")
			tab4.pyplot()
            
		with tab5:
			tab5.header("Generating Word Cloud for Positive Sentiments")
			positive_reviews = Scores_Uber_Review['sentence'][Scores_Uber_Review["Polarity"] == 'Positive']
			positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(positive_reviews))
			plt.figure(figsize = (10, 7))
			plt.title("Positive Sentiment - Wordcloud")
			plt.imshow(positive_wordcloud, interpolation="bilinear")
			plt.axis("off")
			tab5.pyplot()
            
   
		with tab6:
			tab6.header("Generating Word Cloud for Negative Sentiments")
			Negative_reviews = Scores_Uber_Review['sentence'][Scores_Uber_Review["Polarity"] == 'Negative']
			negative_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(Negative_reviews))
			plt.figure(figsize = (10, 7))
			plt.title("Negative Sentiment - Wordcloud")
			plt.imshow(negative_wordcloud, interpolation="bilinear")
			plt.axis("off")
			tab6.pyplot()
            

#Doing "N-Gram Analysis"
	elif option == "N-Gram Analysis":
		start_time = time.time()
		tab1, tab2,tab3, tab4, tab5, tab6 = st.tabs(["Stop Words", "Bi-Grams", "Review_Words",  "Stemmer Comparison", "DTM Matrix", "TF-IDF Matrix"])
      
		bi_grams = Uber_reviews['Review'].apply(lambda row: list(token_ngrams(row,2)))
		Uber_reviews["Review_words"] = Uber_reviews["Review"].apply(lambda x: ' '.join([w for w in x if w not in (stopwords_eng)]))
        
		bigrams = []
		bigrams = list(itertools.chain(*bi_grams))
		bigram_counts = Counter(bigrams)
		bigram_counts.most_common(20)
		# Stemming using the libraries PorterStemmer
		for i in range(len(Uber_reviews['Review'])):
			words  = nltk.tokenize.WhitespaceTokenizer().tokenize(Uber_reviews['Review'][i])
		df = pd.DataFrame()
		df['OriginalWords'] = pd.Series(words)
		#porter's stemmer
		porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
		df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
		#SnowBall stemmer
		snowballStemmedWords = [nltk.stem.SnowballStemmer("english").stem(word) for word in words]
		df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)     
        
		# Generating word level DTM matrix
		Cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		tf = Cv.fit_transform(Uber_reviews["Review_words"])
		dtm_Uber_reviews = pd.DataFrame(tf.toarray())
		tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		tfidf_Uber_reviews = pd.DataFrame(tfidf.fit_transform(Uber_reviews["Review_words"]).toarray(),columns=tfidf.get_feature_names())
		st.write("Time taken to do N-Gram Analysis in seconds:")
		st.write(time.time() - start_time)
        
		with tab1:
			tab1.header("Generating Stop Words")
			Uber_reviews["Review_words"] = Uber_reviews["Review"].apply(word_tokenize)
			tab1.write(Uber_reviews["Review_words"])

		with tab2:
			tab2.header("Printing Bi-Grams")
			tab2.write(bi_grams)
			tab2.write("Length Of Bi-Grams:")
			tab2.write(len(bi_grams))
         

		with tab3:
			tab3.header("Review Words")
			tab3.write(Uber_reviews["Review_words"])
            

		with tab4:
			tab4.header("Stemmer Comparison")
			tab4.write(df)
            
		with tab5:
			tab5.header("Generating DTM Matrix")
			tab5.write(dtm_Uber_reviews.head())
            
		with tab6:
			tab6.header("Generating TF-IDF Matrix")
			tab6.write(tfidf_Uber_reviews.head())
            
#Calling Functions from SupervisedLearning
	# 4. Supervised Learning phase: Rubber meets road now. Run a regression (or classification, if you prefer) of review ratings against the text features you have collected in the previous step. Use any regression or classification method you want to. OLS regression is easy to run and interpret, and is hence preferable
	elif option == "Supervised Learning":
		start_time = time.time()
# split the Data into training, testing and validation datasets 
		X_train, X_test, y_train, y_test = model_selection.train_test_split(Uber_reviews['Review'], Uber_reviews['Rating'],test_size =0.2)

# After splitting into train and test set, get the validation set from the train set
		X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train,test_size =0.25)


		Uber_reviews_Logi = LogisticRegression(max_iter=15000,random_state=0, solver='liblinear')

		dtm_tf_valid, dtm_idf_valid = series2dtm(X_valid, min_df1=5, ngram_range1=(1,2), top_n=200)
		dtm_tf_train, dtm_idf_train = series2dtm(X_train, min_df1=5, ngram_range1=(1,2), top_n=len(dtm_tf_valid.columns))

		Uber_reviews_Logi.fit(dtm_tf_train, y_train)

		y_valid_pred = Uber_reviews_Logi.predict(dtm_tf_valid)

# Accuracy on Validation set
		valid_score =accuracy_score(y_valid,y_valid_pred)
		st.write("Accuracy score for Validation Data: ")
		st.write(valid_score)
                 
		dtm_tf_test, dtm_idf_test = series2dtm(X_test, min_df1=5, ngram_range1=(1,2), top_n=200)
# Accuracy on Test set
		y_test_pred = Uber_reviews_Logi.predict(dtm_tf_test)
		Test_score =accuracy_score(y_test,y_test_pred)
		st.write("Accuracy score for Test Data: ")
		st.write(Test_score)

		Uber_reviews_Logi = LogisticRegression(max_iter=15000,random_state=0, solver='liblinear')

		dtm_tf_valid, dtm_idf_valid = series2dtm(X_valid, min_df1=5, ngram_range1=(1,2), top_n=200)
		dtm_tf_train, dtm_idf_train = series2dtm(X_train, min_df1=5, ngram_range1=(1,2), top_n=len(dtm_idf_valid.columns))

		Uber_reviews_Logi.fit(dtm_idf_train, y_train)

		y_valid_pred = Uber_reviews_Logi.predict(dtm_idf_valid)

		# Accuracy on Validation set

		score =accuracy_score(y_valid,y_valid_pred)
		print("Accuracy score: %.4f\n" % score)

# Accuracy on Test set for TF-IDF

		y_test_pred = Uber_reviews_Logi.predict(dtm_idf_test)

		score =accuracy_score(y_test,y_test_pred)
		print("Accuracy score: %.4f\n" % score)
		st.write("Time taken to do Supervised Learning in seconds:")
		st.write(time.time() - start_time)

         
            










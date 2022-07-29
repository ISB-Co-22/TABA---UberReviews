import streamlit as st 
st.set_page_config(layout="wide")
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
from sklearn import *
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
                                  ["Sentiment Analysis"])
                                 
#Calling Functions from DataPreProcessing
	dh=MissingValues(Uber_reviews) # To check any missing values in the data , Calling From DataPreProcessing.py    
	Uber_reviews = DataCleanUp(Uber_reviews) #CleanUp Data    , Calling From DataPreProcessing.py   
	Uber_reviews = Uber_reviews.applymap(lambda s:s.lower() if type(s) == str else s) # Convert to Lower Case
    
	SentAn_Uber_Review = Uber_reviews['Review']    
	Scores_Uber_Review = vader_corpus_func(SentAn_Uber_Review) 
          
 #Doing sentiment Analysis
	if option == "Sentiment Analysis":
		start_time = time.time()
		tab1, tab2,tab3, tab4, tab5 = st.tabs(["Uber Review Score", "WordCloud", "Bi-Grams", "DTM & TF-IDF", "Supervised Learning"])
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
		time_taken = format(time.time() - start_time,".2f")
		st.write("Time taken to do Sentiment Analysis data in seconds:", time_taken)


		with tab1: 
			tab1.subheader("Uber Reviews Score - After Data Pre-Processing & Sentiment Analysis") 
          
			tab1.write(Scores_Uber_Review.head())  

		with tab2:
			tab2.header("Generating Word Cloud")
			col1, col2 = st.columns(2)
			col1.write = ("Positive Word")
			positive_reviews = Scores_Uber_Review['sentence'][Scores_Uber_Review["Polarity"] == 'Positive']
			positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(positive_reviews))
			plt.figure(figsize = (10, 7))
			plt.title("Positive Sentiment - Wordcloud")
			plt.imshow(positive_wordcloud, interpolation="bilinear")
			plt.axis("off")
			col1.pyplot() 
			col2.write = ("Negative Word")
			positive_reviews = Scores_Uber_Review['sentence'][Scores_Uber_Review["Polarity"] == 'Negative']
			positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(positive_reviews))
			plt.figure(figsize = (10, 7))
			plt.title("Negative Sentiment - Wordcloud")
			plt.imshow(positive_wordcloud, interpolation="bilinear")
			plt.axis("off")
			col2.pyplot()


      

 #Doing "N-Gram Analysis"

		start_time = time.time()
		#tab1, tab2,tab3, tab4, tab5, tab6 = st.tabs(["Stop Words", "Bi-Grams", "Review_Words",  "Stemmer Comparison", "DTM Matrix", "TF-IDF Matrix"])
		Uber_reviews["Review_words"] = Uber_reviews["Review"].apply(word_tokenize)
		bi_grams = Uber_reviews['Review_words'].apply(lambda row: list(token_ngrams(row,2)))
		Uber_reviews["Review_words"] = Uber_reviews["Review"].apply(lambda x: ' '.join([w for w in x if w not in (stopwords_eng)]))
        
		bigrams = []
		bigrams = list(itertools.chain(*bi_grams))
		bigram_counts = Counter(bigrams)
		bigram_counts.most_common(20)
		with tab3:
			tab3.header("Bi-Grams - Most common 10")
			tab3.dataframe(bigram_counts.most_common(10))
            
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
		vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=200, stop_words={'english'})  
		vects = vect.fit_transform(Uber_reviews["Review"])
		td = pd.DataFrame(vects.todense()).iloc[:5]  
		td.columns = vect.get_feature_names()
		term_document_matrix = td.T
		term_document_matrix.columns = ['Doc '+str(i) for i in range(1, 6)]
		term_document_matrix['count'] = term_document_matrix.sum(axis=1)

		term_document_matrix = term_document_matrix.sort_values(by ='count',ascending=False)[:25] 

        
        
		# Generating word level TF-IDF matrix  
		vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=200, stop_words={'english'}) 
		vects = vect.fit_transform(Uber_reviews["Review"])
		td = pd.DataFrame(vects.todense()).iloc[:5]  
		td.columns = vect.get_feature_names()
		TFIDF_matrix = td.T
		TFIDF_matrix.columns = ['Doc '+str(i) for i in range(1, 6)]
		TFIDF_matrix['count'] = TFIDF_matrix.sum(axis=1)

		TFIDF_matrix = TFIDF_matrix.sort_values(by ='count',ascending=False)[:25]        
      
		time_taken = format(time.time() - start_time,".2f")
		st.write("Time taken to do N-Gram Analysis in seconds:",time_taken)
       
            
		with tab4:
			col1, col2 = st.columns(2)
			col1.header("Generating DTM Matrix")
			col1.bar_chart(term_document_matrix['count'])
			col2.header("Generating TF-IDF Matrix")
			col2.bar_chart(TFIDF_matrix['count'])
            


            
#Calling Functions from SupervisedLearning
	# 4. Supervised Learning phase: Rubber meets road now. Run a regression (or classification, if you prefer) of review ratings against the text features you have collected in the previous step. Use any regression or classification method you want to. OLS regression is easy to run and interpret, and is hence preferable
		with tab5: 
			tab5= tab5.header("Supervised Learning Accuracy Scores")
			start_time = time.time()
			# split the Data into training, testing and validation datasets 
			X_train, X_test, y_train, y_test = model_selection.train_test_split(Uber_reviews['Review'], Uber_reviews['Rating'],test_size =0.2)

			# After splitting into train and test set, get the validation set from the train set
			X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train,test_size =0.25)


			Uber_reviews_Logi = LogisticRegression(max_iter=15000,random_state=0, solver='liblinear')

			dtm_tf_valid, dtm_idf_valid = series2dtm(X_valid, min_df1=5, ngram_range1=(1,2))
			dtm_tf_train, dtm_idf_train = series2dtm(X_train, min_df1=5, ngram_range1=(1,2))

			Uber_reviews_Logi.fit(dtm_tf_train, y_train)

			y_valid_pred = Uber_reviews_Logi.predict(dtm_tf_valid)

        # Accuracy on Validation set
			valid_score =accuracy_score(y_valid,y_valid_pred)
			#print("Accuracy score: %.4f\n" % score)
			#st.metric("Accuracy score Validation Set for DTM Uni-Gram: ", valid_score)
        
        
			dtm_tf_test, dtm_idf_test = series2dtm(X_test, min_df1=5, ngram_range1=(1,2), top_n=200)
			# Accuracy on Test set using Unigram and DTM
			y_test_pred = Uber_reviews_Logi.predict(dtm_tf_test)

			score_dtm_unigram =accuracy_score(y_test,y_test_pred)
			#st.metric("Accuracy score for DTM Uni-Gram: ", score_dtm_unigram)

        #I  am here
			# Trying the TF-IDF for training and prediction

			dtm_tf_valid, dtm_idf_valid = series2dtm(X_valid, min_df1=5, ngram_range1=(1,2), top_n=200)
			dtm_tf_train, dtm_idf_train = series2dtm(X_train, min_df1=5, ngram_range1=(1,2), top_n=200)

			Uber_reviews_Logi.fit(dtm_idf_train, y_train)

			y_valid_pred = Uber_reviews_Logi.predict(dtm_idf_valid)

			# Accuracy on Validation set
			valid_score =accuracy_score(y_valid,y_valid_pred)
			#st.metric("Accuracy score Validation Set for TF-IDF Uni-Gram: ", valid_score)
        
        
			# Accuracy on Test set for TF-IDF using Unigram
			y_test_pred = Uber_reviews_Logi.predict(dtm_idf_test)
			score_tfidf_unigram = accuracy_score(y_test,y_test_pred)
			#st.metric("Accuracy score for TF-IDF Uni-Gram: ", score_tfidf_unigram)
        
        
			#Bi-Gram
			dtm_tf_test_bigram, dtm_idf_test_bigram = series2dtm(X_test, min_df1=5, ngram_range1=(2,2), top_n=200)
			dtm_tf_valid_bigram, dtm_idf_valid_bigram = series2dtm(X_valid, min_df1=2, ngram_range1=(2,2), top_n=200)
			dtm_tf_train_bigram, dtm_idf_train_bigram = series2dtm(X_train, min_df1=2, ngram_range1=(2,2), top_n=len(dtm_tf_test_bigram.columns))
			Uber_reviews_Logi.fit(dtm_tf_train_bigram, y_train)

			# Accuracy on Test set using Bi-grams and DTM
			y_test_pred = Uber_reviews_Logi.predict(dtm_tf_test_bigram)
			score_dtm_bigram =accuracy_score(y_test,y_test_pred)
			#st.metric("Accuracy score for DTM Bi-Gram: ", score_dtm_bigram)

            
           
			# Accuracy on Test set using Bi-grams and TF-IDF

			y_test_pred = Uber_reviews_Logi.predict(dtm_idf_test_bigram)
			score_tfidf_bigram =accuracy_score(y_test,y_test_pred)
			#st.metric("Accuracy score for TF-IDF Bi-Gram: ", score_tfidf_bigram)
			st.subheader("Accuracy Score:")       
			df = pd.DataFrame({"Accuracy Test": ["DTM","TF-IDF"],
                 "Uni-Gram": [score_dtm_unigram,score_tfidf_unigram], 
                   "Bi-Gram": [score_dtm_bigram, score_tfidf_bigram]}
                  )
 

			st.dataframe(df)

        
			Scores_Uber_Review = vader_corpus_func(SentAn_Uber_Review)
			sent_score_pred = Scores_Uber_Review[['doc_index','sentence','compound']].copy()
			sent_score_pred['Rating'] = Uber_reviews['Rating'].to_numpy()
			#sent_score_pred
			# split the Data into training, testing and validation datasets 
			X_train, X_test, y_train, y_test = model_selection.train_test_split(sent_score_pred['compound'], sent_score_pred['Rating'],test_size =0.2)
			X_train = np.array(X_train).reshape(-1,1)
			X_test = np.array(X_test).reshape(-1,1)
			Uber_reviews_sent_Logi = LogisticRegression(random_state=0)
			Uber_reviews_sent_Logi.fit(X_train.reshape(-1,1), y_train)
			y_test_sent_pred = Uber_reviews_sent_Logi.predict(X_test)
			score_sent =accuracy_score(y_test,y_test_sent_pred)

			st.write("Accuracy score for Sentiment Score:")
			st.write( format(score_sent,".2f"))
			time_taken = format(time.time() - start_time,".2f")
			st.write("Time taken to do Supervised Learning in seconds:", time_taken)



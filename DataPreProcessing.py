import streamlit as st 
import time
import pandas as pd
import re

st.set_option('deprecation.showPyplotGlobalUse', False)


def MissingValues(dh):
	data = dh.isna().sum()
	return data 


def DataCleanUp(Uber_reviews):
# Cleaning the text
# Cleaning the text
	for row in Uber_reviews.index:
		Uber_reviews.loc[row,'Review'] = re.sub(r"(@\[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", Uber_reviews.loc[row,'Review'])
	return Uber_reviews 


def ChangeCase(Uber_reviews):
# changing the case of the text
	Uber_reviews=Uber_reviews.applymap(lambda s:s.lower() if type(s) == str else s)
	return Uber_reviews 
       

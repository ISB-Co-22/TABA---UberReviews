import streamlit as st 
import time
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data(FILE_ADDRESS):
	data = pd.read_csv(FILE_ADDRESS,encoding='cp1252')
	return data 

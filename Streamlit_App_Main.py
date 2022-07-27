import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')
DATA_URL = ('https://raw.githubusercontent.com/ISB-Co-22/TABA-UberReviews/main/uber_reviews_itune.csv')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(100)
st.write(data)

import fileB
fileB.my_func()

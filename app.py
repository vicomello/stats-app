import streamlit as st
import numpy as np
import pandas as pd

sd = st.slider('sd')
data = np.random.normal(0, sd, 1000)

st.subheader('Distribution as a function of sd')
hist_values = np.histogram(data, bins=50)[0]
st.bar_chart(hist_values)
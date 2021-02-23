#%%

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

#%% config

st.set_page_config(
    page_title="Regression",
    layout="wide",
    initial_sidebar_state="auto",
)

#%% title

st.title("Interpreting Regressions")
st.write(
    "Use the sliders to change the intercept and beta of the regression line"
)
#%% side select box

add_selectbox = st.sidebar.selectbox("Learn more about...", ("a", "b", "c"))

#%% sliders

range_ = [-10.0, 10.0]
intercept = st.slider('intercept', *range_)
beta = st.slider('beta',*range_)

#%% defining linear regression based on slider input
x = list(range(-10,11))

ylist=list()
if intercept != -11:
    for i in x:    
        y = intercept + beta*float(i)
        ylist.append(y)
d = {'x': x, 'y': ylist}
chart_data = pd.DataFrame(data=d)

#%% creating graph

c = alt.Chart(chart_data).mark_line().encode(
    x='x',
    y=alt.Y('y', scale=alt.Scale(domain=(-100,100)))
)

st.altair_chart(c, use_container_width=True)
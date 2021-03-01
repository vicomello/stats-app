#%%

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

#%%

st.set_page_config(
    page_title="T test",
    layout="wide",
    initial_sidebar_state="auto",
)

#%% title

st.title("Interpreting One Sample t-test")
st.write(
    "Use the slider to change the t value"
)
#%% side select box

add_selectbox = st.sidebar.selectbox("Learn more about...", ("a", "b", "c"))

#%% setting slider

t_values = [-5,5]
t_value = st.slider('t value', *t_values, value=0)

#%% defining lines

x_data = list(range(-10,11))
y1_data = list() # This is the line for the inputed t
y2_data = list() # This is the reference line, at t = 0

for i in x_data:
    y1_data.append(t_value)
    y2_data.append(0)

data = {'t': y1_data, 'reference': y2_data, 'x': list(range(-10,11))}
df = pd.DataFrame(data)

#%% drawing visualization

chartdata = df.melt('x')

chart = alt.Chart(chartdata).mark_line().encode(
    x='x',
    y=alt.Y('value', scale=alt.Scale(domain=(-10,10))),
    color='variable'
)
st.altair_chart(chart, use_container_width=True)


#st.line_chart(dataframe)


#c = alt.Chart(dataframe).mark_line().encode(
#    x=list(range(0,11))
#    y=alt.Y('y', scale=alt.Scale(domain=(-10,10)))
#)

#st.altair_chart(c, use_container_width=True)
# %%

# y_1 y_2 - both the t of both lines to be drawn
# x is same for both

# %%

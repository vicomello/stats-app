#%%

import streamlit as st
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import norm

#%% config

st.set_page_config(
    page_title="your_title",
    layout="wide",
    initial_sidebar_state="auto",
)

#%% title

st.title("Normal distribution")
st.write(
    "Use the sliders to change the mean and standard deviation of the normal distribution. The figure is interactive, so you can hover over it, zoom in/out, or drag it around."
)

#%% side select box

add_selectbox = st.sidebar.selectbox("Learn more about...", ("a", "b", "c"))

#%% slider

mu_values = [-100.0, 100.0]
mu_values.append(float(np.median(mu_values)))

sigma_values = [1.0, 50.0, 15.0]

mu = st.slider(f"Mean of distribution", *mu_values)
st.write("Mean: ", mu)
sigma = st.slider(f"Standard deviation of distribution", *sigma_values)
st.write("Standard deviation: ", sigma)


#%% define normal distribution based on slider input

x = np.linspace(mu - 100 * sigma, mu + 100 * sigma, 2000)
y = norm.pdf(x, mu, sigma)
df = pd.DataFrame({"x": x, "y": y})

#%% plot and display

st.write("#")  # add break

# https://stackoverflow.com/questions/52223358/rename-tooltip-in-altair
fig = (
    alt.Chart(df)
    .mark_line(size=4, color="#51127c")
    .encode(
        x=alt.X(
            "x",
            scale=alt.Scale(domain=(mu_values[0], mu_values[1])),
            axis=alt.Axis(title="Mean"),
        ),
        y=alt.Y("y", axis=alt.Axis(title="Normal probability density")),
        tooltip=[alt.Tooltip("x", title="Mean"), alt.Tooltip("y", title="Density")],
    )
    .interactive()
)
st.altair_chart(fig, use_container_width=True)

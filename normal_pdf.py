#%%

import streamlit as st
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import norm

#%% slider

mu_values = [-100.0, 100.0]
mu_values.append(float(np.median(mu_values)))

sigma_values = [1.0, 100.0, 15.0]

mu = st.slider(f"Mean of distribution", *mu_values)
st.write("Mean: ", mu)
sigma = st.slider(f"Standard deviation of distribution", *sigma_values)
st.write("Standard deviation: ", sigma)


#%% define normal distribution based on slider input

x = np.linspace(mu - 100 * sigma, mu + 100 * sigma, 2000)
y = norm.pdf(x, mu, sigma)
df = pd.DataFrame({"x": x, "y": y})

#%% plot and display

# https://stackoverflow.com/questions/52223358/rename-tooltip-in-altair
fig = (
    alt.Chart(df)
    .mark_line(size=4, color="#51127c")
    .encode(
        x=alt.X(
            "x",
            scale=alt.Scale(domain=(mu_values[0] * 4, mu_values[1] * 4)),
            axis=alt.Axis(title="Mean"),
        ),
        y=alt.Y("y", axis=alt.Axis(title="Normal probability density")),
        tooltip=[alt.Tooltip("x", title="Mean"), alt.Tooltip("y", title="Density")],
    )
    .interactive()
)
st.altair_chart(fig, use_container_width=True)

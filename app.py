#%%

import streamlit as st
import numpy as np
import pandas as pd

#%%
# sd = st.slider("sd")
# data = np.random.normal(0, sd, 1000)

# st.subheader("Distribution as a function of sd")
# hist_values = np.histogram(data, bins=50)[0]
# st.bar_chart(hist_values, width=10, height=300)

# %%

import ttest_onesample

ttest_onesample.run_ttest_onesample()


# stats.norm.pdf(x, mu, sigma)
# pd.DataFrame()
# ax.plot()
# fig.show()  ##shows figure on console

# draw linear function in graph
# people can decide intercept and slope
# B0 and B1
# go to altair

#%%

import numpy as np
import pandas as pd
import streamlit as st

import home
import regression
import multiple_regression
import ttest_independent
import ttest_onesample
import utils

#%% Page

st.set_page_config(
    page_title="General linear model app",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="random",
)
#%% Sidebar that displays the pages

# The page names and the page files
PAGES = {
    # "Start here": home,
    "One-sample t-test": ttest_onesample,
    "Independent-samples t-test": ttest_independent,
    "Simple linear regression": regression,
    # "Multiple linear regression": multiple_regression,
}

# Displaying the selection box
st.sidebar.markdown("## What's next?")
selection = st.sidebar.selectbox("", list(PAGES.keys()))
page = PAGES[selection]
page.main()

# %%

st.sidebar.markdown("# ")
st.sidebar.markdown(f"###### Made by {utils.authors()}")
st.sidebar.markdown("## ")
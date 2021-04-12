#%%

import numpy as np
import pandas as pd
import streamlit as st

import home
import regression
import ttest_independent
import ttest_onesample
import utils

#%% Page

st.set_page_config(
    page_title="",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="random",
)
#%% Sidebar that displays the pages

# The page names and the page files
PAGES = {
    # "Start here": home,
    # "One-sample t-test": ttest_onesample,
    "Independent-samples t-test": ttest_independent,
    "Regression": regression,
}

# Displaying the selection box
st.sidebar.title("Where to begin?")
selection = st.sidebar.selectbox("", list(PAGES.keys()))
page = PAGES[selection]
page.main()

# %%

st.sidebar.markdown("Some more text here.")

st.sidebar.markdown(f"Made by {utils.authors()}.")

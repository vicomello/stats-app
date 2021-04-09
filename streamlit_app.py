#%%
import numpy as np
import pandas as pd
import streamlit as st

import home
import regression
import ttest_independent
import ttest_onesample

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
    "Goals of this app": home,
    "One-sample t-test": ttest_onesample,
    "Independent-samples t-test": ttest_independent,
    "regression": regression,
}

# Displaying the selection box
st.sidebar.title("See other tests:")
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.main()

# %%

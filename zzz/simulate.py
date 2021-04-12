#%%

import os
import altair as alt
from vega_datasets import data
import numpy as np
import pandas as pd
import utils

import streamlit as st

pd.set_option(
    "display.max_rows",
    10,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)
np.set_printoptions(
    edgeitems=5,
    linewidth=300,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

#%% config

st.set_page_config(
    page_title="Simulate",
    layout="wide",
    initial_sidebar_state="auto",
)

#%% title

st.title("Simulating data")
st.write("Simulating and visualizing data.")


#%% simulate and plot data

n1 = 100
mu1 = 0
sd1 = 30

n2 = 40
mu2 = 130
sd2 = 15

x1 = utils.rand_norm_fixed(n1, mu1, sd1)
x2 = utils.rand_norm_fixed(n2, mu2, sd2)
print(np.std(x1))
print(np.mean(x1))

c1 = ["control"] * len(x1)
c2 = ["drug"] * len(x2)

x1.extend(x2)
c1.extend(c2)

df = pd.DataFrame({"condition": c1, "bp": x1})


np.array([1., 2., 3.]) * 3.0 + 3.0


#%% plot and display

st.write("#")

# https://datavizpyr.com/stripplot-with-altair-in-python/

fig = (
    alt.Chart(df)
    .mark_circle(size=20)
    .encode(
        y=alt.Y(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        x=alt.X("bp:Q"),
        color=alt.Color("condition:N", legend=None),
        column=alt.Row(
            "condition:N",
            # header=alt.Header(
            #     labelAngle=0,
            #     titleOrient="top",
            #     labelOrient="bottom",
            #     labelAlign="center",
            #     labelPadding=10,
            # ),
        ),
    )
    .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
    .interactive()
    .properties(width=300, height=200)
)

st.altair_chart(fig, use_container_width=False)

st.write("hey")

#%%

fig2 = (
    alt.Chart(df)
    # .transform_fold(
    #     ["Trial A", "Trial B", "Trial C"], as_=["Experiment", "Measurement"]
    # )
    .mark_area(opacity=0.5, interpolate="step")
    .encode(
        alt.X("bp:Q", bin=alt.Bin(maxbins=40)),
        alt.Y("count()", stack=None),
        alt.Color("condition:N"),
    )
    .interactive()
    .properties(width=200)
)
st.altair_chart(fig2, use_container_width=True)

#%%

fig3 = (
    alt.Chart(df)
    .mark_tick(color="#51127c", size=5)
    .encode(x="bp:Q", y="condition:O", color=alt.Color("condition:N", legend=None))
    .interactive()
    .properties(height=200, width=500)
)
st.altair_chart(fig3, use_container_width=False)


#%%

# fig4 = (
#     alt.Chart(df)
#     .transform_density(density="bp", groupby=["condition"])
#     .mark_area()
#     .encode(alt.X("bp:Q"), alt.Y("density:Q"), alt.Row("condition:N"))
#     .properties(width=300, height=50)
# )
# # fig4
# st.altair_chart(fig4, use_container_width=False)

#%%

fig5 = (
    alt.Chart(df)
    .mark_boxplot(opacity=0.2)
    .encode(x="bp:Q", y="condition:N")
    .properties(height=200, width=500)
)
# fig5

st.altair_chart(fig5, use_container_width=False)


#%%

fig6 = fig3 + fig5

st.altair_chart(fig6, use_container_width=False)

#%%

st.write("#")


source = data.movies.url

stripplot = (
    alt.Chart(source, width=40)
    .mark_circle(size=8)
    .encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y("IMDB_Rating:Q"),
        color=alt.Color("Major_Genre:N", legend=None),
        column=alt.Column(
            "Major_Genre:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
    )
    .transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)


st.altair_chart(stripplot, use_container_width=False)


#%%

# see
# https://altair-viz.github.io/gallery/scatter_with_layered_histogram.html
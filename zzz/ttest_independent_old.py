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
    precision=3,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

#%% config

st.set_page_config(
    page_title="t-test independent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#%% title

st.title("t-test: independent samples")
st.write("Use the sliders to choose your values.")
st.write("# ")


#%% side select box

add_selectbox = st.sidebar.selectbox(
    "Learn more about other tests",
    ("t-test: one sample", "t-test: independent samples"),
)

#%% sliders

conds = ["control", "treatment"]

# specify slider ranges and start points
n1_values = [1, 50, 5]
n2_values = [1, 50, 5]

mu1_values = [-200.0, 200.0, -50.0]
mu2_values = [-200.0, 200.0, 50.0]

sigma1_values = [1.0, 100.0, 15.0]
sigma2_values = [1.0, 100.0, 15.0]

# coding for regression
coding1_values = [-1.5, 1.5, 0.0]
coding2_values = [-1.5, 1.5, 1.0]


col1, col2, col3, col4 = st.beta_columns([1, 1, 0.3, 3])
with col1:
    st.markdown("**Sample size**")
    n1 = st.slider(f"Control", *n1_values)
    n2 = st.slider(f"Treatment", *n1_values)
    st.write(n1, ", ", n2)
    st.write("### ")

    st.markdown("**Coding (x-axis positions)**")
    coding1 = st.slider(f"Control", *coding1_values)
    coding2 = st.slider(f"Treatment", *coding2_values)
    st.write(coding1, ", ", coding2)

with col2:
    st.markdown("**Means (y-axis positions)**")
    mu1 = st.slider(f"Control", *mu1_values)
    mu2 = st.slider(f"Treatment", *mu2_values)
    st.write(mu1, ", ", mu2)
    st.write("### ")

    st.markdown("**Standard deviations**")
    sd1 = st.slider(f"Standard deviation (SD) - control", *sigma1_values)
    sd2 = st.slider(f"Standard deviation (SD) - treatment", *sigma2_values)
    st.write(sd1, ",", sd2)


#%% simulate data

x1 = utils.rand_norm_fixed(n1, mu1, sd1)
x2 = utils.rand_norm_fixed(n2, mu2, sd2)
c1 = [conds[0]] * len(x1)
c2 = [conds[1]] * len(x2)
x1.extend(x2)
c1.extend(c2)
conds_recoded = np.array([coding1, coding2])

df = pd.DataFrame({"condition": c1, "profit": x1})
df["profit"] = np.round(df["profit"], 3)
df.loc[df["condition"] == conds[0], "code"] = conds_recoded[0]
df.loc[df["condition"] == conds[1], "code"] = conds_recoded[1]

# mean
df["condition_mean_profit"] = np.round(
    df.groupby("condition").transform(np.mean)["profit"], 2
)

df = df[["condition", "code", "profit", "condition_mean_profit"]]

#%%

m1, m2 = np.round(df.groupby("condition").mean()["profit"], 2)


#%% show data

st.write("#")  # add break
# st.dataframe(df)

#%% plot figure

conds_recoded_x_domain = [conds_recoded[0] - 0.5, conds_recoded[1] + 0.5]
if conds_recoded_x_domain[0] >= 0.0:
    conds_recoded_x_domain[0] = -0.5
# TODO automatically determine y and x domains

y_domain = [df["profit"].min() * 0.9, df["profit"].max() * 1.1]

fig1 = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x=alt.X("code:Q", scale=alt.Scale(domain=conds_recoded_x_domain)),
        y=alt.Y("profit", scale=alt.Scale(domain=y_domain)),
        color="condition",
        tooltip=["condition", "profit"],
    )
    .properties(width=520, height=350)
    .interactive()
)
# fig1

fig2 = (
    alt.Chart(df)
    .mark_line(color="gray")
    .encode(x="code:Q", y=alt.Y("mean(profit)", title=""))
    # .properties(width=500)
)
fig3 = fig1 + fig2
# fig3

# TODO add horizontal line for intercept
# TODO add mean point for each condition

#%%

vline = pd.DataFrame([{"vline": 0}])
fig4 = (
    alt.Chart(vline)
    .mark_rule(color="black", opacity=1.0, strokeDash=[3, 5])
    .encode(x="vline:Q")
)

fig5 = fig3 + fig4

# st.altair_chart(fig5)

#%%

# st.write("### Simulated data and general linear model representation")
# st.write("###")

# col1, col2 = st.beta_columns(2)
# with col1:


with col4:
    st.write("### General linear model")
    st.altair_chart(fig5, use_container_width=True)
    st.write("##### Simulated data")
    st.write("##### ")
    st.dataframe(df, height=290)

# with col2:
#     st.dataframe(df, height=290)

# if st.button("save dataframe"):
#     open("df.csv", "w").write(df.to_csv())

#%%

eq2 = r"y = c + mx"
st.latex(eq2)

eq3 = r"y = b_{0} + b_{1}x"
st.latex(eq3)


delta_x = np.diff(conds_recoded)[0]
delta_y = m2 - m1

eq1 = r"""
    slope = b_1 = 
    \frac{{\Delta} y}{{\Delta} x} = 
    \frac{y_{treatment} - y_{control}}{x_{treatment} - x_{control}} = 
    \frac{m2 - m1}{x2 - x1} = 
    \frac{delta_y}{delta_x} = 
    result
    """
eq1 = eq1.replace("m2", f"{m2}").replace("m1", f"{m1}")
eq1 = eq1.replace("x2", f"{conds_recoded[1]}").replace("x1", f"{conds_recoded[0]}")
eq1 = eq1.replace("delta_y", f"{delta_y}").replace("delta_x", f"{delta_x}")
eq1 = eq1.replace("result", f"{delta_y / delta_x}")

st.latex(eq1)

eq4 = f"y = b_0 + {delta_y/delta_x} x"
st.latex(eq4)

eq5 = f"{m2} = b_0 + {delta_y/delta_x}({conds_recoded[1]})"
st.latex(eq5)

eq6 = f"{m2} = b_0 + {delta_y/delta_x * conds_recoded[1]}"
st.latex(eq6)

eq7 = f"b_0 = {m2} - {delta_y/delta_x * conds_recoded[1]}"
st.latex(eq7)

eq8 = f"b_0 = {m2 - delta_y/delta_x * conds_recoded[1]} = intercept"
st.latex(eq8)
#%%

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import utils
import pingouin as pg

# %matplotlib inline
# plt.style.use(['fast'])
# plt.ion()

#%% config

st.set_page_config(
    page_title="One-sample t-test",
    layout="wide",
    initial_sidebar_state="collapsed",
)

add_selectbox = st.sidebar.selectbox(
    "Learn more about...", ("Independent-samples t-test", "b", "c")
)

#%% title and description

st.title("One-sample t-test")
st.markdown(
    "We use the **one-sample t-test** when we have a **sample** (i.e., a set of data points we've collected) and we want to know whether the **mean of our sample** is **different from a specific value**."
)

st.markdown(
    "For example, on 10 different days, you rate and record your own happiness using a scale that ranges from -10 to 10 (-10: miserable, 0: neutral, 10: ecstatic). You can use the one-sample t-test to see whether you're generally happy or sad. That is, you test whether you mean happiness over the last 10 days is bigger or smaller than 0 (neutral)."
)

# %% container for examples

my_expander = st.beta_expander("Click for more examples")
with my_expander:
    st.markdown("A few more short examples here.")
    st.markdown("####")

# %% equations

st.markdown("####")
st.markdown(
    "The one-sample t-test [general linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) is following linear equation:"
)
st.latex("y_i = b_0 + \epsilon_i")

st.markdown(
    "where $y_i$ are the data points, $b_0$ is the intercept (i.e., the value of $y$ when $x$ is 0), $\epsilon_i$ is the residual associated with data point $y_i$."
)
st.markdown(
    "Note that there is only $b_0$ (intercept) in the equation. There aren't $b_1$, $b_2$ and so on—there are no slopes (i.e., the slopes are 0). Thus, the one-sample t-test is just a linear equation with a **horizontal line** that crosses the y-intercept at $b_0$, which is the mean of the sample."
)
st.markdown("#")


#%% make columns/containers

col1, cola, col2, colb, col3 = st.beta_columns(
    [0.8, 0.05, 0.8, 0.05, 1.3]
)  # ratios of widths

#%% create sliders

slider_n_params = [
    "Number of data points (sample size, N)",  # label
    2,  # min
    200,  # max
    10,  # start value
    1,  # step
]
slider_mean_params = [
    "Sample mean",
    -50.0,
    50.0,
    5.00,
    0.1,
]
slider_sd_params = [
    "Standard deviation (SD) or 'spread'",
    0.1,
    50.0,
    13.8,
    0.1,
]

with col2:
    st.markdown("Drag the sliders to simulate data")
    n = st.slider(*slider_n_params)
    slider_n_params[3] = n
    mean = st.slider(*slider_mean_params)
    slider_mean_params[3] = mean
    sd = st.slider(*slider_sd_params)
    slider_sd_params[3] = sd
    st.write("N, mean, SD: ", n, ",", mean, ",", sd)

# %% create/show dataframe

df1 = pd.DataFrame({"Happiness": utils.rand_norm_fixed(n, mean, sd), "Rating": 0})
df1["Happiness"] = df1["Happiness"].round(1)
df1["i"] = np.arange(1, df1.shape[0] + 1)
df1["Mean"] = df1["Happiness"].mean().round(1)
df1["Residual"] = df1["Happiness"] - df1["Mean"]

# t-test
res = pg.ttest(df1["Happiness"], 0)
df1["d"] = res["cohen-d"][0]

with col3:
    st.markdown("Sample data (each row is one simulated data point $i$)")
    st.write(df1[["i", "Happiness", "Mean", "Residual"]].round(2))


#%% generate and draw data points

x_domain = [-0.1, 0.1]
y_max = (np.ceil(df1["Happiness"].max()) + 2.0) * 1.3
y_min = (np.floor(df1["Happiness"].min()) - 2.0) * 1.3
y_domain = [y_min, y_max]

fig1 = (
    alt.Chart(df1)
    .mark_circle(size=89, color="#57106e", opacity=0.8)
    .encode(
        x=alt.X(
            "Rating:Q",
            scale=alt.Scale(domain=x_domain),
            axis=alt.Axis(grid=False, title="", tickCount=2, labels=False),
        ),
        y=alt.Y(
            "Happiness:Q",
            scale=alt.Scale(domain=y_domain),
            axis=alt.Axis(grid=False, title="Happiness (y)", titleFontSize=13),
        ),
        tooltip=["Happiness", "i", "Mean", "Residual"],
    )
    # .properties(width=144, height=377)
    # .properties(title="General linear model")
    .interactive()
)

#%% horizontal line for b0 mean

hline_b0 = pd.DataFrame(
    {"b0 (mean)": [mean], "N": [n], "SD": [sd], "Model": f"y = {mean} + e"},
)

fig2 = (
    alt.Chart(hline_b0)
    .mark_rule(size=3, color="#f98e09")
    .encode(
        y=alt.Y("b0 (mean):Q", axis=alt.Axis(title="")),
        tooltip=["Model", "b0 (mean)", "SD", "N"],
    )
    .interactive()
    # .properties(width=233, height=377)
)

#%% intercepts

fig3 = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(size=1, color="#000004", opacity=0.8, strokeDash=[3, 3])
    .encode(y=alt.Y("y:Q", axis=alt.Axis(title="")))
    # .properties(width=233, height=377)
)

#%% combine figures

finalfig = fig3 + fig2 + fig1
finalfig.configure_axis(grid=False)
finalfig.title = hline_b0["Model"][0]
with col1:
    st.markdown("General linear model")
    st.altair_chart(finalfig, use_container_width=True)

#%% show t test results

st.markdown("####")
res_list = []
res_list.append(res["dof"].round(0)[0])  # df
res_list.append(res["T"].round(2)[0])
p = np.round(res["p-val"][0], 3)
if p < 0.001:
    ptext = "p < "
    pval = 0.001
else:
    ptext = "p = "
    pval = p
res_list.append(pval)
bf = np.round(float(res["BF10"][0]), 2)
if bf > 10000:
    bftext = "Bayes factor > "
    bf = 10000
else:
    bftext = "Bayes factor = "
res_list.append(bf)
res_list.append(res["cohen-d"].round(2)[0])
res_list.append(res["power"].round(2)[0] * 100)

st.markdown("### One-sample t-test results")

st.write(
    "t(",
    res_list[0],
    ") = ",
    res_list[1],
    f", {ptext}",
    pval,
    f", {bftext}",
    bf,
    "Cohen's d effect size = ",
    res_list[4],
    "power = ",
    res_list[5],
    "%",
)


# %% container derivation

st.markdown("####")
my_expander = st.beta_expander("Click to see derivation")
with my_expander:
    st.markdown("Latex equations here")
    st.markdown("####")

# %% show code

st.markdown("####")
my_expander = st.beta_expander("Click to see Python and R code")
with my_expander:
    st.markdown("R: `lm(y ~ 1)`  # linear model with only intercept term")
    st.markdown("R: `t.test(y, mu = 0)`  # t-test against 0")
    st.markdown("Python: `pingouin.ttest(y, 0)`  # t-test against 0")
    st.markdown("####")


# %% container prompts

st.markdown("####")
my_expander = st.beta_expander("Test your intuition")
with my_expander:
    st.markdown("questions here")
    st.markdown("####")

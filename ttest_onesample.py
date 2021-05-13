#%%

import altair as alt
import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st

import utils

#%% config


def main():

    #%% create stuff for side bar

    #%% create sliders
    slider_n_params = [
        "Sample size (N, or no. of data points)",  # label
        2,  # min
        50,  # max
        14,  # start value
        1,  # step
        "%f",  # format
    ]
    slider_mean_params = ["Sample mean (average)", -4.0, 4.0, 2.0, 0.1, "%f"]
    slider_sd_params = [
        'Standard deviation (SD) or "spread"',
        0.1,
        10.0,
        7.0,
        0.1,
        "%f",
    ]

    st.sidebar.markdown("Change values to simulate new data")
    st.sidebar.markdown("#####")
    n = st.sidebar.slider(*slider_n_params)
    slider_n_params[3] = n
    st.markdown("#####")
    mean = st.sidebar.slider(*slider_mean_params)
    slider_mean_params[3] = mean
    st.sidebar.write("Mean = $b_0$ = intercept = ", mean)
    st.markdown("#####")
    sd = st.sidebar.slider(*slider_sd_params)
    slider_sd_params[3] = sd

    # %% create/show dataframe

    df1 = pd.DataFrame({"Happiness": utils.rand_norm_fixed(n, mean, sd), "Rating": 0})
    df1["i"] = np.arange(1, df1.shape[0] + 1)
    df1["Happiness"] = df1["Happiness"].round(2)
    df1["Mean"] = df1["Happiness"].mean().round(2)
    df1["Residual"] = df1["Happiness"] - df1["Mean"]
    df1["Residual"] = df1["Residual"].round(2)
    for i in df1.itertuples():
        df1.loc[
            i.Index, "Model"
        ] = f"{i.Happiness:.2f} = {i.Mean:.2f} + {i.Residual:.2f}"

    # t-test
    res = pg.ttest(df1["Happiness"], 0)
    df1["d"] = res["cohen-d"][0]

    # %% create figures

    x_domain = [-0.1, 0.1]
    y_domain = [-30, 30]
    fig_height = 377

    fig1 = (
        alt.Chart(df1)
        .mark_circle(size=(89 / np.sqrt(n)) * 2, color="#3b528b", opacity=0.8)
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
            tooltip=["i", "Happiness", "Mean", "Residual", "Model"],
        )
        .properties(height=377)
        # .properties(title="General linear model")
        .interactive()
    )

    #%% horizontal line for b0 mean

    hline_b0 = pd.DataFrame(
        {"b0 (mean)": [mean], "N": [n], "SD": [sd], "Model": f"y = {mean} + e"},
    )

    fig2 = (
        
        alt.Chart(hline_b0)
        .mark_rule(size=3, color="#51127c")
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
        .mark_rule(size=0.5, color="#000004", opacity=0.5, strokeDash=[3, 3])
        .encode(y=alt.Y("y:Q", axis=alt.Axis(title="")))
        # .properties(height=fig_height)
    )

    # %% violin

    fig5 = (
        alt.Chart(df1)
        .transform_density(
            density="Happiness", as_=["Happiness", "density"], bandwidth=2.0
        )
        .mark_area(orient="horizontal", opacity=0.3, color="#3b528b")
        .encode(
            alt.X(
                "density:Q",
                title="",
                stack="zero",
                impute=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            alt.Y(
                "Happiness:Q",
            ),
        )
    )

    #%% combine figures

    finalfig = fig3 + fig2 + fig5 + fig1
    finalfig.configure_axis(grid=False)
    finalfig.height = fig_height
    # finalfig.title = hline_b0["Model"][0]

    #%% app title and description

    st.markdown("# One-sample t-test")
    st.markdown(
        "We use the **one-sample t-test** when we have a **sample** (i.e., a set of $N$ data points we've collected) and we want to know whether the **mean of our sample** is **different from a specific value** (it can be any value, but let's assume it's 0 here)."
    )
    st.markdown("### Are you happy or sad?")
    st.write(
        "You want to know whether you're happy or sad in general, so you recorded your own happiness, once a day, for",
        n,
        "days. Scores > 0 means you felt happy on those days; scores < 0 means you felt sad. Below, each dot is your happiness for one day (hover over the dots for more information), and the mean (average) across all days is ",
        mean,
        ".",
    )

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        st.write(
            "Your scores for each day (`i`) are in the `Happiness` column, ",
            n,
            "values/rows, one for each day, `i`). The mean (average) of the scores are in the `Mean` column. `Residual` is `Happiness` minus `Mean` for each value/row.",
        )
        # format dataframe output
        fmt = {
            "i": "{:.0f}",
            "Happiness": "{:.1f}",
            "Mean": "{:.1f}",
            "Residual": "{:.1f}",
        }
        dfcols = ["i", "Happiness", "Mean", "Residual"]  # cols to show
        st.dataframe(df1[dfcols].style.format(fmt), height=233)

    #%% show figure

    _, col_fig, _ = st.beta_columns([0.1, 0.5, 0.1])  # hack to center figure
    with col_fig:
        st.altair_chart(finalfig, use_container_width=False)

    #%% show t test results

    my_expander = st.beta_expander("Click here to see t-test results")
    with my_expander:
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
        res_list.append(res["power"].round(2)[0])

        st.write(
            "t(",
            res_list[0],
            ") = ",
            res_list[1],
            f", {ptext}",
            pval,
            f", {bftext}",
            bf,
            ", ",
            "Cohen's d effect size = ",
            res_list[4],
            ", statistical power = ",
            res_list[5],
        )

        eq1 = r"t = \frac{\bar{y} - \mu_{0}}{SE_{\bar{y}}} = \frac{MEAN - 0}{STERR} = TVAL"
        eq1 = (
            eq1.replace("MEAN", str(mean))
            .replace("STERR", str(np.round(sd / np.sqrt(n), 2)))
            .replace("TVAL", str(res_list[1]))
        )
        st.latex(eq1)
        st.latex(r"SE_{\bar{y}} = \frac{SD}{\sqrt{N}}")

        st.write(
            """
        * $t$: t-statistic (degrees of freedom in brackets)
        * $p$: probability probability of obtaining results at least as extreme as what we have observed assuming the null hypothesis is correct
        * Bayes factor: relative evidence for the tested model over the null model
        * Cohen's d effect size: a common effect size metric 
        * statistical power: probability of detecting an effect assuming it exists
        """
        )

    # %% container derivation

    # %% equations

    st.markdown("##### ")
    eq1 = "y_i = b_0 + \epsilon_i"
    st.latex(eq1)
    st.latex(eq1.replace("b_0", str(mean)))

    st.markdown(
        "where $y_i$ are the data points ($y_1, y_2, ... y_{n-1}, y_n$), $b_0$ is the intercept (the value at which the line crosses the $$y$$-axis), $\epsilon_i$ is the residual associated with data point $y_i$."
    )
    st.write(
        "This model says that the **best predictor of $y_i$ is $b_0$**, which is the **intercept** or the **mean** (or average) of all the data points ($b_0$ =",
        mean,
        "). So if you want to predict any value $y_i$, use the mean of your sample.",
    )
    st.write(
        "This model is very simple (i.e., predict every $y_i$ by assuming every $y_i$ equals the mean value of the sample: $y_i = b_0$), so it might make bad predictions. For example, your mean happiness is 2.0 ($b_0 = 2.0$), but on day 13, your score was 27 ($y_{13} = 27$). That is, $27 = 2.0 + \epsilon_{13}$, where $\epsilon_{13} = 25$ is the residual for that day—it's how wrong the model was."
    )
    st.markdown(
        "Note that there is only $b_0$ (intercept) in the equation. There aren't $b_1$, $b_2$ and so on—there are no slopes (i.e., the slopes are 0). Thus, the one-sample t-test is just a linear model or equation with a **horizontal line** that crosses the y-intercept at $b_0$, which is the mean of the sample."
    )

    # %% show python/R code

    my_expander = st.beta_expander("Click to see Python and R code")
    with my_expander:
        st.markdown(
            "The one-sample t-test is equivalent to a linear regression with only the intercept. Thus, the following functions/methods in Python and R will return the same results."
        )

        st.markdown(
            "Python: `pingouin.ttest(y, 0)`  # t-test against 0 (can be any other value)"
        )
        st.markdown(
            "Python: `scipy.stats.ttest_1samp(y, 0)`  # t-test against 0 (can be any other value)"
        )
        st.markdown(
            'Python: `statsmodels.formula.api.ols(formula="y ~ 1", data=dataframe).fit()`  # linear model with only intercept term'
        )
        st.markdown("####")

        st.markdown("R: `t.test(y, mu = 0)`  # t-test against 0")
        st.markdown("R: `lm(y ~ 1)`  # linear model with only intercept term")

# %%

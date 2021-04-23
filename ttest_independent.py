#%%

import altair as alt
import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st

import utils


#%% config
def main():

    #%% sidebar

    #%% create sliders for humans

    slider_n_params = [
        " ",  # label
        2,  # min
        50,  # max
        25,  # start value
        1,  # step
        "%f",  # format
    ]
    slider_mean_params = [
        "  ",
        -6.0,
        6.0,
        2.0,
        0.1,
        "%f",  # format
    ]
    slider_sd_params = [
        "",
        0.1,
        10.0,
        6.0,
        0.1,
        "%f",  # format
    ]

    #%% create sliders for martians

    slider_n2_params = [
        "",  # label
        2,  # min
        50,  # max
        30,  # start value
        1,  # step
        "%f",  # format
    ]
    slider_mean2_params = [
        "",
        -6.0,
        6.0,
        3.0,
        0.1,
        "%f",  # format
    ]
    slider_sd2_params = [
        "",
        0.1,
        10.0,
        4.0,
        0.1,
        "%f",  # format
    ]

    # sidebar headers
    sidebar_headers = st.sidebar.beta_columns(2)  # ratios of widths
    with sidebar_headers[0]:
        st.markdown("### Humans")
    with sidebar_headers[1]:
        st.markdown("### Martians")

    # sidebar columns - sample size
    st.sidebar.markdown("#### ")
    st.sidebar.markdown("##### Sample size (N)")
    col11, col12 = st.sidebar.beta_columns(2)  # ratios of widths
    with col11:
        # st.markdown("Humans")
        n = st.slider(*slider_n_params)
        slider_n_params[3] = n
    with col12:
        # st.markdown("Martians")
        n2 = st.slider(*slider_n2_params)

    # sidebar columns - mean
    st.sidebar.markdown("##### Mean (average)")
    col21, col22 = st.sidebar.beta_columns(2)  # ratios of widths
    with col21:
        mean = st.slider(*slider_mean_params)
        slider_mean_params[3] = mean
    with col22:
        mean2 = st.slider(*slider_mean2_params)
        slider_mean2_params[3] = mean2

    # sidebar columns - sd
    st.sidebar.markdown("##### Standard deviation (SD)")
    col31, col32 = st.sidebar.beta_columns(2)  # ratios of widths
    with col31:
        sd = st.slider(*slider_sd_params)
        slider_sd_params[3] = sd
    with col32:
        sd2 = st.slider(*slider_sd2_params)
        slider_sd2_params[3] = sd2

    code1_params = [" ", -1.0, 1.0, 0.0, 0.01]
    code2_params = ["  ", -1.0, 1.0, 1.0, 0.01]

    # sidebar container - advanced settings
    st.sidebar.write("#### Advanced options")
    st.sidebar.text("Change code for each group")
    col41, col42 = st.sidebar.beta_columns(2)
    with col41:
        code1 = st.slider(*code1_params)
    with col42:
        code2 = st.slider(*code2_params)
    center_code = st.sidebar.checkbox("Center predictors")

    #%% simulate data

    df1 = pd.DataFrame(
        {
            "Happiness": utils.rand_norm_fixed(n, mean, sd),
            "Species": "Human",
            "Code": code1,
            "Group_mean": mean,
        }
    )
    df2 = pd.DataFrame(
        {
            "Happiness": utils.rand_norm_fixed(n2, mean2, sd2),
            "Species": "Martian",
            "Code": code2,
            "Group_mean": mean2,
        }
    )
    df_all = pd.concat([df1, df2], axis=0)
    df_all["i"] = np.arange(1, df_all.shape[0] + 1)
    df_all["Happiness"] = df_all["Happiness"].round(2)
    df_all["Mean"] = df_all["Happiness"].mean().round(2)
    df_all["Code_centered"] = df_all["Code"] - df_all["Code"].mean()
    df_all["Residual"] = df_all["Happiness"] - df_all["Mean"]
    df_all["Residual"] = df_all["Residual"].round(2)

    # create tooltip for plot (Model: ...)
    for i in df_all.itertuples():
        df_all.loc[
            i.Index, "Model"
        ] = f"{i.Happiness:.2f} = {i.Mean:.2f} + {i.Residual:.2f}"

    # group mean
    df_mean = (
        df_all.groupby("Species").mean().reset_index()[["Species", "Happiness", "Code"]]
    )

    #%% ttest and linear regression

    X = df_all[["Code"]]
    if center_code:
        X = df_all[["Code_centered"]]

    y = df_all["Happiness"]
    df_results = pg.linear_regression(X, y, add_intercept=True)
    b0 = df_results.at[0, "coef"]
    b1 = df_results.at[1, "coef"]
    model = f"y = {np.round(b0, 2)} + {np.round(b1, 2)}x1 + e"

    #%% Centering the coding based on the checkbox
    if center_code == True:
        x_coding = "Code_centered:Q"
    else:
        x_coding = "Code:Q"

    # %% plot

    x_domain = [-1.15, 1.15]
    y_domain = [-30, 30]
    fig_height = 377

    fig1 = (
        alt.Chart(df_all)
        .mark_circle(size=(89 / np.sqrt(n)) * 2, color="#57106e", opacity=0.5)
        .encode(
            x=alt.X(
                x_coding,
                scale=alt.Scale(domain=x_domain),
                axis=alt.Axis(grid=False, title="", tickCount=2),
            ),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False, title="Happiness (y)", titleFontSize=13),
            ),
            color=alt.Color("Species"),
            tooltip=["i", "Happiness", "Mean", "Residual", "Model"],
        )
        .interactive()
        .properties(height=377, width=477)
    )

    #%% horizontal line for b0 mean

    hline_b0 = pd.DataFrame(
        {
            "b0 (mean)": [df_all["Happiness"].mean().round(2)],
            "N": [n + n2],
            "SD": [sd],
            "Model": f"y = {mean} + e",
        },
    )

    #  fig4 = (
    #      alt.Chart(hline_b0)
    #      .mark_rule(size=3.4, color="#bc3754")
    #      .encode(
    #          y=alt.Y("b0 (mean):Q", axis=alt.Axis(title="")),
    #          tooltip=["Model", "b0 (mean)", "SD", "N"],
    #      )
    #      .interactive()
    #      .properties(width=233, height=377)
    #  )

    #%% intercepts

    fig4 = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(size=0.5, color="#000004", opacity=0.5, strokeDash=[3, 3])
        .encode(y=alt.Y("y:Q", axis=alt.Axis(title="")))
        .properties(height=fig_height)
    )

    fig5 = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(size=0.5, color="#000004", opacity=0.5, strokeDash=[3, 3])
        .encode(x=alt.X("x:Q", axis=alt.Axis(title="")))
        .properties(height=fig_height)
    )

    # %% violin
    # https://altair-viz.github.io/gallery/violin_plot.html

    # fig5 = (
    #     alt.Chart(df_all)
    #     .transform_density(
    #         density="Happiness", as_=["Happiness", "density"], bandwidth=2.0
    #     )
    #     .mark_area(orient="horizontal", opacity=0.3, color="#f98e09")
    #     .encode(
    #         alt.X(
    #             "Species:N",
    #             title="",
    #             stack="zero",
    #             impute=None,
    #             axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
    #         ),
    #         alt.Y(
    #             "Happiness:Q",
    #         ),
    #         color=alt.Color("Species"),
    #     )
    #     .properties(height=fig_height)
    # )

    #%% fig 2: the means for each sample and a line connecting them

    # df_means = df_all.groupby("Species")

    # plot means
    fig2 = (
        alt.Chart(df_mean)
        .mark_point(filled=True, size=144)
        .encode(
            x=alt.X(
                x_coding,
                scale=alt.Scale(domain=x_domain),
                axis=alt.Axis(grid=False, title="", tickCount=2),
            ),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False, title="Happiness (y)", titleFontSize=13),
            ),
            color=alt.Color("Species"),
            tooltip=["Happiness", "Code"],
        )
        .interactive()
    )

    # connect means with line
    fig3 = (
        alt.Chart(df_mean)
        .mark_line(color="black")
        .encode(
            x=alt.X(
                x_coding,
                scale=alt.Scale(domain=x_domain),
                # axis=alt.Axis(grid=False, title="", tickCount=2, labels=False),
                axis=alt.Axis(grid=False, title="", tickCount=2),
            ),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False, title="Happiness (y)", titleFontSize=13),
            ),
        )
        .interactive()
    )

    #%% violin plots
    fig6 = (
        alt.Chart(df1)
        .transform_density(
            density="Happiness", as_=["Happiness", "density"], bandwidth=2.0
        )
        .mark_area(orient="horizontal", opacity=0.3, color="#5F64C8")
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
        .properties(height=fig_height)
    )

    fig7 = (
        alt.Chart(df2)
        .transform_density(
            density="Happiness", as_=["Happiness", "density"], bandwidth=2.0
        )
        .mark_area(orient="horizontal", opacity=0.3, color="#f98e09")
        .encode(
            alt.X(
                "density:Q",
                title="",
                stack="zero",
                impute=None,
                axis=alt.Axis(labels=False, values=[1], grid=False, ticks=True),
            ),
            alt.Y(
                "Happiness:Q",
            ),
        )
        .properties(height=fig_height)
    )

    #%% title and description
    st.title("Independent-samples t-test")
    st.markdown(
        "We use the **independent-samples t-test** when we have **two unrelated (independent) samples** (i.e., two distinct datasets with $N$ data points each) and we want to know whether the mean of the two samples are different from each other."
    )
    st.markdown("### Who's happier—humans or Martians?")
    st.markdown(
        "Humans and Martians disagree on who is happier, so each species gathered a bunch of their own members and recorded their happiness for a day. Scores > 0 means above-average happiness; scores < 0 means below-average happiness. Their happiness scores are shown below."
    )

    expander_df = st.beta_expander("Click here to see the simulated data")
    with expander_df:
        st.markdown(
            "In the dataframe below, your scores for each day (`i`) are in the `Happiness` column (25 values/rows, one for each day, `i`). The mean (average) of the scores are in the `Mean` column. `Residual` is `Happiness` minus `Mean` for each value/row."
        )

        st.dataframe(
            # df_all[["i", "Happiness", "Mean", "Residual"]].style.format("{:.1f}"),
            df_all[["i", "Species", "Code", "Happiness", "Mean", "Residual"]],
            height=233,
        )

    # st.markdown("### Interactive app")
    st.markdown("###### ")
    # st.markdown(
    #     "$y_i = b_0 + b_1 x_1 + \epsilon_i$ is the [general linear model](https://en.wikipedia.org/wiki/General_linear_model) for the independent-samples t-test (more explanation below)."
    # )

    # more text
    # To develop an intuition, change the values in the sliders below, explore the (simulated) data in the dataframe (click any column name to sort by that column), or hover over the data points on the interactive figure to understand this model. To reset to the default values, refresh the page."

    # %% show figurs

    finalfig = fig1 + fig2 + fig3 + fig4 + fig5 + fig6 + fig7
    finalfig.configure_axis(grid=False)
    finalfig.title = model
    st.altair_chart(finalfig, use_container_width=False)

    #%% show dataframe

    #%% calculate t test to show to them
    res = pg.ttest(df1["Happiness"], df2["Happiness"])
    # df1["d"] = res["cohen-d"][0]

    #%% show t test results (optional)

    expander_ttest = st.beta_expander(
        "Click here to see detailed independent-samples t-test results"
    )
    with expander_ttest:
        st.markdown(
            "The values in green will update as you change the slider values above."
        )
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

    st.markdown("## General linear model")
    st.markdown(
        "The one-sample t-test [general linear model](https://en.wikipedia.org/wiki/General_linear_model) is following linear equation:"
    )
    eq1 = "y_i = b_0 + b_1 x_1 + \epsilon_i"
    eq1 = eq1.replace("b_0", str(np.round(b0, 2)))
    eq1 = eq1.replace("b_1", str(np.round(b1, 2)))
    st.latex(eq1)

    st.markdown(
        "where $y_i$ are the data points ($y_1, y_2, ... y_{n-1}, y_n$), $b_0$ is the intercept (i.e., the value of $y$ when $x$ is 0), $\epsilon_i$ is the residual associated with data point $y_i$. Simulated $y_i$ (happiness scores for each day) and $\epsilon_i$ (residuals for each day) are shown in the dataframe above."
    )
    st.write(
        "This model says that the **best predictor of $y_i$ is $b_0$**, which is the **intercept** or the **mean** (or average) of all the data points ($b_0$ =",
        mean,
        "). So if you want to predict any value $y_i$, use the mean of your sample.",
    )
    st.write(
        "Since this model is very simple (i.e., predict every $y_i$ by assuming every $y_i$ equals the mean value of the sample: $y_i = b_0$), it can result in bad predictions. For example, your mean happiness is 2.0 ($b_0 = 2.0$), but on day 13, your score was 27 ($y_{13} = 27$). That is, $27 = 2.0 + \epsilon_{13}$, where $\epsilon_{13} = 25$ is the residual for that day—it's how wrong the model was."
    )
    st.markdown(
        "Note that there is only $b_0$ (intercept) in the equation. There aren't $b_1$, $b_2$ and so on—there are no slopes (i.e., the slopes are 0). Thus, the one-sample t-test is just a linear model or equation with a **horizontal line** that crosses the y-intercept at $b_0$, which is the mean of the sample."
    )
    # %% show code

    my_expander = st.beta_expander("Click to see Python and R code")
    with my_expander:
        st.markdown(
            "The one-sample t-test is equivalent to a linear regression with only the intercept. Thus, the following functions/methods in Python and R will return the same results."
        )

        st.markdown(
            "Python: `pingouin.ttest(y_group1, y_group2)`  # t-test against 0 (can be any other value)"
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
        st.markdown(
            "R: `lm(y ~ condition)`  # linear model - intercept term is implicitly added"
        )
        st.markdown(
            "R: `lm(y ~ 1 + condition)`  # same as above - explicitly including intercept term"
        )

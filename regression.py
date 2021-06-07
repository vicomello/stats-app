#%%

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pingouin as pg

import utils

#%% config
def main():

    slider_n_params = [
        "Sample size (no. of data points)",  # label
        2,  # min
        100,  # max
        8,  # start value
        2,  # step
        "%f",  # format
    ]

    slider_b0_params = [
        "b0 (intercept)",  # label
        -30.0,  # min
        30.0,  # max
        15.0,  # start value
        0.1,  # step
        "%f",  # format
    ]
    slider_b1_params = [
        "b1 (slope)",
        -3.0,
        3.0,
        -0.9,
        0.1,
        "%f",  # format
    ]
    slider_noise_params = [
        "Noise (standard deviation)",
        0.0,
        50.0,
        7.5,
        0.1,
        "%f",  # format
    ]

    n = st.sidebar.slider(*slider_n_params)
    slider_n_params[3] = n

    b0 = st.sidebar.slider(*slider_b0_params)
    slider_b0_params[3] = b0

    b1 = st.sidebar.slider(*slider_b1_params)
    slider_b1_params[3] = b1

    noise = st.sidebar.slider(*slider_noise_params)
    slider_noise_params[3] = noise

    np.random.seed(int(n + b0 + b1 + noise))  # hack: freeze state

    st.sidebar.markdown("#### Rescale predictor (hunger)?")
    predictor_scale = st.sidebar.radio(
        "", ("Use raw values", "Mean-center", "Z-score"), key="x"
    )

    st.sidebar.markdown("#### Rescale outcome (happiness)?")
    outcome_scale = st.sidebar.radio(
        "", ("Use raw values", "Mean-center", "Z-score"), key="y"
    )

    #%% defining linear regression
    df = pd.DataFrame({"Hunger": utils.simulate_x(n, [-20, 40])})
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Happiness"] = utils.simulate_y(df[["Hunger"]], np.array([b0, b1]), noise)
    df["Mean_Happiness"] = df["Happiness"].mean()
    df["Happiness_Centered"] = df["Happiness"] - df["Happiness"].mean()
    df["Happiness_zscore"] = (df["Happiness"] - df["Mean_Happiness"]) / df[
        "Happiness"
    ].std()
    df["Mean_Hunger"] = df["Hunger"].mean()
    df["Hunger_Centered"] = df["Hunger"] - df["Hunger"].mean()
    df["Hunger_zscore"] = (df["Hunger"] - df["Mean_Hunger"]) / df["Hunger"].std()
    df["b0"] = b0
    df["b1"] = b1

    X = "Hunger:Q"
    x_domain = [-100, 100]  # figure x domain
    y_domain = [-100, 100]
    if predictor_scale == "Mean-center":
        X = "Hunger_Centered:Q"
    elif predictor_scale == "Z-score":
        X = "Hunger_zscore:Q"
        x_domain = [i / 20 for i in x_domain]
    x_col = X.replace(":Q", "")

    # TODO scale outcome/response (same as if/else statement above)

    # TODO need to think about how to refactor the code below (too repetitive) (depends on how we want to present latex)

    lm = pg.linear_regression(df[[x_col]], df["Happiness"], add_intercept=True)
    b0, b1 = lm["coef"].round(2)

    lm_raw = pg.linear_regression(df[["Hunger"]], df["Happiness"], add_intercept=True)
    b0_raw, b1_raw = lm_raw["coef"].round(2)

    lm_zX = pg.linear_regression(
        df[["Hunger_zscore"]], df["Happiness"], add_intercept=True
    )
    b0_zX, b1_zX = lm_zX["coef"].round(2)

    lm_zXY = pg.linear_regression(
        df[["Hunger_zscore"]], df["Happiness_zscore"], add_intercept=True
    )
    b0_zXY, b1_zXY = lm_zXY["coef"].round(2)

    df["Predicted_Happiness"] = b0 + b1 * df[x_col]
    df["Residual"] = df["Happiness"] - df["Predicted_Happiness"]

    # TODO pingouin correlation

    #%% title

    st.title("Simple linear regression")
    st.markdown(
        "We use a linear regression model when we want to understand how the change in a variable Y influences on another variable X."
    )
    st.markdown("### How does hunger relate to happiness?")
    st.write("Lorem Ipsum dolor.")

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        # format dataframe output
        fmt = {
            "Hunger": "{:.2f}",  # TODO also show Hungercenterd, hungerzscore
            "Happiness": "{:.2f}",
            "Predicted_Happiness": "{:.2f}",
            "Residual": "{:.2f}",
        }
        dfcols = [
            "Hunger",
            "Happiness",
            "Predicted_Happiness",
            "Residual",
        ]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=233)

    #%% interactive dots for model
    for i in df.itertuples():
        df.loc[
            i.Index, "Model"
        ] = f"{i.Happiness:.2f} = ({b0} + {b1} * {i.Hunger:.2f}) + {i.Residual:.2f}"
        df

    # TODO make it interactive; change color scheme; add x/y labels
    fig_main = (
        alt.Chart(df)
        .mark_circle(size=55, color="#3b528b", opacity=0.8)
        .encode(
            x=alt.X(X, scale=alt.Scale(domain=x_domain), axis=alt.Axis(grid=False)),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False),
            ),
            tooltip=[
                "Hunger",
                "Happiness",
                "Predicted_Happiness",
                "Residual",
                "b1",
                "b0",
                "Model",
            ],
        )
        .properties(height=377, width=377)
    )
    fig_main.interactive()  # https://github.com/altair-viz/altair/issues/2159

    # TODO make line interactive (show tooltip model)
    fig_regline = fig_main.transform_regression(
        x_col, "Happiness", extent=[-300, 300]
    ).mark_line(size=3, color="#b73779")
    fig_regline.interactive()

    #%% Horizontal line

    fig_horizontal = (
        alt.Chart(pd.DataFrame({"Y": [0]}))
        .mark_rule(size=2, color="#5ec962", opacity=0.8, strokeDash=[5, 5])
        .encode(y=alt.Y("Y:Q", axis=alt.Axis(title="")))
        .properties(height=377)
    )

    #%% Vertical Line

    fig_vertical = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(size=2, color="#5ec962", opacity=0.8, strokeDash=[5, 5])
        .encode(x=alt.X("x:Q", axis=alt.Axis(title="")))
        .interactive()
        # .properties(height=fig_height)
    )

    df_intercept = pd.DataFrame({"x": [0], "y": [b0], "b0 (intercept)": [b0]})
    fig_b0dot = (
        alt.Chart(df_intercept)
        .mark_point(size=89, fill="#51127c", color="#51127c")
        .encode(x="x:Q", y="y:Q", tooltip=["b0 (intercept)"])
        .interactive()
    )

    #%% Drawing plot

    finalfig = fig_horizontal + fig_vertical + fig_regline + fig_b0dot + fig_main
    _, col_fig, _ = st.beta_columns([0.15, 0.5, 0.1])  # hack to center figure
    with col_fig:
        st.altair_chart(finalfig, use_container_width=False)

    my_expander = st.beta_expander("Click here to show/hide regression results")
    with my_expander:
        fmt = {
            "coef": "{:.2f}",
            "se": "{:.2f}",
            "T": "{:.2f}",
            "pval": "{:.3f}",
            "r2": "{:.2f}",
            "adj_r2": "{:.2f}",
        }
        dfcols = ["names"]  # cols to show
        dfcols += fmt.keys()
        st.dataframe(lm[dfcols].style.format(fmt), height=233)

    #%% Writing GLM
    st.markdown("##### ")
    eq1 = "y_i = b_0 + b_1 x_i + \epsilon_i"
    st.latex(eq1)
    eq2 = (
        eq1.replace("b_0", str(b0_raw))
        .replace("b_1", str(b1_raw))
        .replace("y_i", "happiness_i")
        .replace("x_i", "\ hunger_i")
        .replace("\epsilon_i", "residual_i")
    )
    st.latex(eq2)
    eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"
    st.latex(eq3)

    eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"  # TODO replace with beta actual values
    eq3 = eq3.replace(r"\beta_0", f"{b0_zX}").replace(r"\beta_1", f"{b1_zX}")
    st.latex(eq3)

    st.latex("r_{pearson} = some \ number")  # TODO  insert correlation

    # TODO add correlation

    st.markdown(
        "where $x_i$ are the data points ($x_1, x_2, ... x_{n-1}, x_n$), $b_0$ is the intercept (the value at which the line crosses the $$y$$-axis), $b_1$ is the change in Y when you change one unit in x (hunger_i), and $\epsilon_i$ is the residual associated with data point $y_i$ (happiness_i)."
    )
    st.write(
        "This model says that the **best predictor of $y_i$ is $b_1$**, which is the **predictor** or the **independent variable**.",
        "So if you want to predict any value $y_i$, multiplY $b_1$ by its distance from the sun (in AU) and sum the intercept ($b_0$).",
    )

    my_expander = st.beta_expander("Click to show/hide Python and R code")
    with my_expander:
        st.markdown(
            "`X` is the predictor/design matrix and `y` is the outcome/response variable"
        )
        st.markdown("Python: `pingouin.linear_regression(X, y)`")
        st.markdown("Python: `scipy.stats.linregress(X, y)`")
        st.markdown(
            'Python: `statsmodels.formula.api.ols(formula="y ~ X", data=dataframe).fit()`'
        )
        st.markdown("R: `lm(y ~ X)`")


# %%

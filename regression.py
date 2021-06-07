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
        16,  # start value
        2,  # step
        "%f",  # format
    ]

    slider_b0_params = [
        "b0 (intercept)",  # label
        -30.0,  # min
        30.0,  # max
        5.0,  # start value
        0.1,  # step
        "%f",  # format
    ]
    slider_b1_params = [
        "b1 (slope)",
        -1.5,
        1.5,
        1.0,
        0.1,
        "%f",  # format
    ]
    slider_noise_params = [
        "Noise (standard deviation)",
        0.0,
        15.0,
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

    # TODO radio butons instead of separate checkbox (either mean center OR zscore—can't check both at the same time!)
    mean_center_predictor = st.sidebar.checkbox("Mean-center predictor (Age)")
    zscore_predictor = st.sidebar.checkbox("Z-score predictor (Age)")

    # TODO scale outcome/response

    #%% defining linear regression
    df = pd.DataFrame({"Age": utils.simulate_x(n, [5, 60])})
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Happiness"] = utils.simulate_y(
        df[["Age"]], np.array([b0, b1]), residual_sd=noise
    )
    df["Mean_Happiness"] = df["Happiness"].mean()
    df["Mean_Age"] = df["Age"].mean()
    df["Age_Centered"] = df["Age"] - df["Age"].mean()
    df["Age_zscore"] = (df["Age"] - df["Mean_Age"]) / df["Age"].std()

    X = "Age:Q"
    if mean_center_predictor:
        X = "Age_Centered:Q"
        x_label = "Age (mean-centered)"  # for plotting label
    if zscore_predictor:
        X = "Age_zscore:Q"
        x_label = "Age (z-score)"
    x_col = X.replace(":Q", "")

    lm = pg.linear_regression(df[[x_col]], df["Happiness"], add_intercept=True)
    b0, b1 = lm["coef"].round(2)

    df["Predicted_Happiness"] = b0 + b1 * df[x_col]
    df["Residual"] = df["Happiness"] - df["Predicted_Happiness"]

    # TODO pingouin correlation

    #%% title

    st.title("Simple linear regression")
    st.markdown(
        "We use a linear regression model when we want to understand how the change in a variable Y influences on another variable X."
    )
    st.markdown("### How does age relate to happiness?")
    st.write("Lorem Ipsum dolor.")

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        # format dataframe output
        fmt = {
            "Age": "{:.2f}",  # TODO also show Agecenterd, agezscore
            "Happiness": "{:.2f}",
            "Predicted_Happiness": "{:.2f}",
            "Residual": "{:.2f}",
        }
        dfcols = ["Age", "Happiness", "Predicted_Happiness", "Residual"]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=233)

    x_domain = [-100, 100]
    if zscore_predictor:
        x_domain = [i / 20 for i in x_domain]
    y_domain = [-100, 100]

    # TODO make it interactive; change color scheme; add x/y labels
    fig_main = (
        alt.Chart(df)
        .mark_circle(size=55)
        .encode(
            x=alt.X(X, scale=alt.Scale(domain=x_domain), axis=alt.Axis(grid=False)),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False),
            ),
            tooltip=["Age", "Happiness"],
        )
        .properties(height=377, width=377)
    )
    fig_main.interactive()  # https://github.com/altair-viz/altair/issues/2159

    # TODO make line interactive (show model)
    fig_regline = fig_main.transform_regression(
        x_col, "Happiness", extent=[-300, 300]
    ).mark_line()
    fig_regline.interactive()

    #%% Horizontal line

    fig_horizontal = (
        alt.Chart(pd.DataFrame({"Y": [0]}))
        .mark_rule(size=0.5, color="#000004", opacity=0.5, strokeDash=[3, 3])
        .encode(y=alt.Y("Y:Q", axis=alt.Axis(title="")))
        .properties(height=377)
    )

    #%% Vertical Line

    fig_vertical = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(size=0.7, color="#51127c", opacity=0.5, strokeDash=[3, 3])
        .encode(x=alt.X("x:Q", axis=alt.Axis(title="")))
        .interactive()
        # .properties(height=fig_height)
    )

    # TODO make dot interactive
    df_intercept = pd.DataFrame({"x": [0], "y": [b0], "b0 (intercept)": [b0]})
    fig_b0dot = (
        alt.Chart(df_intercept)
        .mark_point(size=89, fill="#57106e", color="#57106e")
        .encode(x="x:Q", y="y:Q", tooltip=["b0 (intercept)"])
        .interactive()
    )

    #%% Drawing plot

    finalfig = fig_main + fig_regline + fig_horizontal + fig_vertical + fig_b0dot
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
    eq1 = "y_i = b_0 + b_1 X_1 + \epsilon_i"
    st.latex(eq1)
    eq2 = eq1.replace("b_0", str(b0)).replace("b_1", str(b1))
    st.latex(eq2)
    eq3 = r"y_i = \beta_0 + \beta_1 X_1 + \epsilon_i"
    st.latex(eq3)

    # TODO add correlation

    st.markdown(
        "where $X_i$ are the data points ($X_1, X_2, ... X_{n-1}, X_n$), $b_0$ is the intercept (the value at which the line crosses the $$Y$$-axis), $b_1$ is the change in Y when you change one unit in X, and $\epsilon_i$ is the residual associated with data point $Y_i$."
    )
    st.write(
        "This model says that the **best predictor of $Y_i$ is $b_1$**, which is the **predictor** or the **independent variable**.",
        "So if you want to predict any value $Y_i$, multiplY $b_1$ by its distance from the sun (in AU) and sum the intercept ($b_0$).",
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

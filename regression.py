#%%

from numpy.lib.shape_base import _replace_zero_by_x_arrays
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import pingouin as pg

import utils

#%% config
def main():

    beta_params = [-1.00, 1.00, 0.25, 0.01]  # min  # maX  # start  # step

    b0 = 10.0
    st.sidebar.markdown("Intercept = 10.0")
    beta_values = [-1.00, -0.5, 0, 0.50, 1.00]
    b1 = st.sidebar.radio("beta", options=beta_values, index=3)  # TODO change to slider

    mean_center_predictor = st.sidebar.checkbox("Mean-center")
    scale_predictor = st.sidebar.checkbox("Z-score predictor (Age)")

    # TODO scale outcome/response

    #%% defining linear regression
    df = pd.DataFrame({"Age": list(range(30, 60))})
    df = pd.DataFrame({"Age": [10, 10, 10, 40, 40, 40]})
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Predicted_Happiness"] = df["Age"] * b1  # FIXME think about how to do this
    df["Residual"] = utils.rand_norm_fixed(df.shape[0], 0, 3)
    df["Happiness"] = b0 + df["Age"] * b1
    df["Mean Happiness"] = np.mean(
        df["Happiness"]
    )  # should be changing age, not happiness (age_mean)
    df["Mean Age"] = df["Age"].mean()
    df["Age Centered"] = df["Age"] - df["Mean Age"]
    df["Age_Scaled"] = df["Age"] / np.std(df["Age"])
    # z-score
    df["Age_Centered_Scaled"] = (df["Age"] - df["Mean Age"]) / df["Age"].std()
    # how to generate a random value for each line?

    # TODO test this
    # utils.rand_norm_fixed(n=df.shape[0], mean=b0 + df["Age"] * b1, sd=3)

    # TODO z-score vs mean-center
    if mean_center_predictor and scale_predictor:
        X = "Age_Centered_Scaled:Q"
        x_values = "Age_Centered_Scaled"
        y_intercept = b0 + b1 * (np.mean(df["Age"]) / np.std(df["Age"]))
    elif mean_center_predictor:
        X = "Age Centered:Q"
        x_values = "Age Centered"
        y_intercept = b0 + (b1 * np.mean(df["Age"]))
    elif scale_predictor:
        X = "Age_Scaled:Q"
        x_values = "Age_Scaled"
        y_intercept = b0 + (b1 * np.std(df["Age"]))
    else:
        X = "Age:Q"
        x_values = "Age"
        y_intercept = b0

    lm = pg.linear_regression(df["Age"], df["Happiness"], add_intercept=True)
    b0, b1 = lm["coef"]
    # TODO get b0 b1 (see tttest)

    # TODO pingouin correlation

    #%% title

    st.title("Interpreting Simple Linear Regressions")
    st.markdown(
        "We use a linear regression model when we want to understand how the change in a variable Y influences on another variable X."
    )
    st.markdown("### How does age relate to happiness?")
    st.write("Lorem Ipsum dolor.")

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        # format dataframe output
        fmt = {
            "Age": "{:.2f}",
            "Happiness": "{:.2f}",
            "Predicted_Happiness": "{:.2f}",
            "Residual": "{:.2f}",
        }
        dfcols = ["Age", "Happiness", "Predicted_Happiness", "Residual"]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=233)

    x_domain = [-80, 80]
    y_domain = [-60, 80]
    # fig_height = 377

    fig1 = (
        alt.Chart(df)
        .mark_circle(size=89)
        .encode(
            x=alt.X(X, scale=alt.Scale(domain=x_domain), axis=alt.Axis(grid=False)),
            y=alt.Y(
                "Happiness:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False),
            ),
        )
    )

    fig2 = fig1.transform_regression(
        x_values, "Happiness", extent=[-80, 80]
    ).mark_line()

    #%% Horizontal line

    fig3 = (
        alt.Chart(pd.DataFrame({"Y": [0]}))
        .mark_rule(size=0.5, color="#000004", opacity=0.5, strokeDash=[3, 3])
        .encode(y=alt.Y("Y:Q", axis=alt.Axis(title="")))
        .properties(height=377)
    )

    #%% Vertical Line

    fig4 = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(size=0.7, color="#51127c", opacity=0.5, strokeDash=[3, 3])
        .encode(x=alt.X("x:Q", axis=alt.Axis(title="")))
        .interactive()
        # .properties(height=fig_height)
    )

    # Yi=10.0+1.0X1+ϵi

    df_intercept = pd.DataFrame({"x": [0], "y": [y_intercept]})

    fig5 = (
        alt.Chart(df_intercept)
        .mark_point(size=60, color="#FF0000")
        .encode(x="x:Q", y="y:Q")
    )

    #%% Drawing plot

    finalfig = fig1 + fig2 + fig3 + fig4 + fig5
    # finalfig = fig1 + fig3 + fig4
    st.altair_chart(finalfig, use_container_width=True)

    my_expander = st.beta_expander("Click here to see regression results")
    with my_expander:
        st.write(lm)

    #%% Writing GLM
    st.markdown("##### ")
    eq1 = "Y_i = b_0 + b_1 X_1 + \epsilon_i"
    st.latex(eq1)
    eq2 = eq1.replace("b_0", str(b0)).replace("b_1", str(b1))
    st.latex(eq2)
    eq3 = r"Y_i = \beta_0 + \beta_1 X_1 + \epsilon_i"
    st.latex(eq3)

    # TODO add correlation

    st.markdown(
        "where $X_i$ are the data points ($X_1, X_2, ... X_{n-1}, X_n$), $b_0$ is the intercept (the value at which the line crosses the $$Y$$-axis), $b_1$ is the change in Y when you change one unit in X, and $\epsilon_i$ is the residual associated with data point $Y_i$."
    )
    st.write(
        "This model says that the **best predictor of $Y_i$ is $b_1$**, which is the **predictor** or the **independent variable**.",
        "So if you want to predict any value $Y_i$, multiplY $b_1$ by its distance from the sun (in AU) and sum the intercept ($b_0$).",
    )

    my_expander = st.beta_expander("Click to see Python and R code")
    with my_expander:
        st.markdown(
            "Python: `pingouin.linear_regression(X, Y)`  # linear regression where X is the predictor and Y the observed variable"
        )
        st.markdown(
            "Python: `scipy.stats.linregress (X, Y)`  # linear regression where X  is the predictor and Y the observed variable"
        )
        st.markdown(
            'PYthon: `statsmodels.formula.api.ols(formula="Y ~ X", data=dataframe).fit()`  # linear model with one predictor'
        )
        st.markdown("####")

        st.markdown("R: `lm(Y ~ X)`  # linear model with one predictor")


# %%

#%%

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import pingouin as pg

import utils

#%% config
def main():
    #st.set_page_config(
    #    page_title="Regression",
    #    layout="wide",
    #    initial_sidebar_state="auto",
    #)
    #%% computations
    range_ = [-10.00, 10.00, 0.00]
    beta_params = [
        -1.00,  # min
        1.00,  # max
        0.25,  #start
        0.01  #step
    ]

    intercept = st.sidebar.slider('intercept', *range_)
    beta = st.sidebar.slider('Beta',*beta_params)

    mean_center = st.sidebar.checkbox("Mean-center")
    scale_data = st.sidebar.checkbox("Scale data")

    #%% defining linear regression
    distance = [0.39, 0.72, 1.00, 1.52, 5.20, 9.53, 19.18, 30.06, 39.53]
    planets =  ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]#distance of the 9 planets from sun in AU
    df = pd.DataFrame({"Distance": distance, "Planet": planets})
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Predicted_Happiness"] = intercept + df["Distance"]*beta
    df["Residual"] = utils.rand_norm_fixed(9,0,1)
    df["Happiness"] = intercept + df["Distance"]*beta + df["Residual"]
    df["Mean"] = np.mean(df["Happiness"])
    #how to generate a random value for each line?
    
    #%% title

    st.title("Interpreting Regressions")
    st.markdown(
        "We use a linear regression model when we want to understand how the change in a variable X influences on another variable Y."
    )
    st.markdown("### How does distance from the sun relate to happiness?")
    st.write(
        "If you move to a planet further to the sun, how much would your degree of happiness change?",
        "With regression models, we want to understand how a one unit change in th predictor variable (in this case, the distance from the sun) influences changes in the observed variable (happiness)",
        "In the example below, distance to the sun is measured in AU (astronomical units). 1 AU = the distance between the sun and planet Earth."
    )

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        # format dataframe output
        fmt = {
            "Planet": "{}",
            "Happiness": "{:.2f}",
            "Predicted_Happiness": "{:.2f}",
            "Residual": "{:.2f}",
        }
        dfcols = ["Planet", "Happiness", "Predicted_Happiness", "Residual"]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=233)

    # _, col_fig, _ = st.beta_columns([0.1, 0.5, 0.1])  # hack to center figure
    # with col_fig:
    #     st.write("hi!")
    #     #st.altair_chart(finalfig, use_container_width=False)
    # #%% sliders
    
    x_domain = [-40, 40]
    y_domain = [-50, 50]
    fig_height = 377
    
    
    fig1 = (
        alt.Chart(df)
        .mark_circle(size=89)
        .encode(
            x=alt.X("Distance:Q",
            scale=alt.Scale(domain=x_domain)),
            y=alt.Y("Happiness:Q",
            scale=alt.Scale(domain=y_domain)),
            color="Planet"
        )
    )
    
    fig2 = fig1.transform_regression('Distance', 'Happiness').mark_line()

    finalfig = fig1 + fig2

    st.altair_chart(finalfig, use_container_width=True)
    
    
    lm = pg.linear_regression(df["Distance"], df["Happiness"], add_intercept=True)
    my_expander = st.beta_expander("Click here to see regression results")
    with my_expander:
        st.write(lm)
        
    #%% Writing GLM
    st.markdown("##### ")
    eq1 = "y_i = b_0 + b_1 x_1 + \epsilon_i"
    st.latex(eq1)
    eq2 = (
        eq1.replace("b_0", str(intercept))
        .replace("b_1", str(beta))
    )
    st.latex(eq2)

    st.markdown(
        "where $y_i$ are the data points ($y_1, y_2, ... y_{n-1}, y_n$), $b_0$ is the intercept (the value at which the line crosses the $$y$$-axis), $b_1$ is the change in Y when you change one unit in X, and $\epsilon_i$ is the residual associated with data point $y_i$."
    )
    st.write(
        "This model says that the **best predictor of $y_i$ is $b_1$**, which is the **predictor** or the **independent variable**.",
        "So if you want to predict any value $y_i$, multiply $b_1$ by its distance from the sun (in AU) and sum the intercept ($b_0$).",
    )

    my_expander = st.beta_expander("Click to see Python and R code")
    with my_expander:
        st.markdown(
            "Python: `pingouin.linear_regression(x, y)`  # linear regression where x is the predictor and y the observed variable"
        )
        st.markdown(
            "scipy.stats.linregress (x, y)`  # linear regression where x is the predictor and y the observed variable"
        )
        st.markdown(
            'Python: `statsmodels.formula.api.ols(formula="y ~ x", data=dataframe).fit()`  # linear model with one predictor'
        )
        st.markdown("####")

        st.markdown("R: `lm(y ~ x)`  # linear model with one predictor")

    #%% defining linear regression based on slider input
    # x = list(range(-10,11))

    # ylist=list()
    # if intercept != -11:
    #     for i in x:    
    #         y = intercept + beta*float(i)
    #         ylist.append(y)
    # d = {'x': x, 'y': ylist}
    # chart_data = pd.DataFrame(data=d)

    #%% creating graph

    # c = alt.Chart(chart_data).mark_line().encode(
    #     x='x',
    #     y=alt.Y('y', scale=alt.Scale(domain=(-100,100)))
    # )

    # st.altair_chart(c, use_container_width=True)
# %%

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
    #    laXout="wide",
    #    initial_sidebar_state="auto",
    #)
    #%% computations
    #range_ = [-10.00, 10.00, 0.00]
    beta_params_humans = [
        -1.00,  # min
        1.00,  # maX
        0.25,  #start
        0.01  #step
    ]
    beta_params_martians = [
        -1.00,  # min
        1.00,  # maX
        0.75,  #start
        0.01  #step
    ]

    intercept = 10.00
    st.sidebar.markdown("Intercept = 10.0")
    beta_humans = st.sidebar.slider('B for Humans',*beta_params_humans)
    beta_martians = st.sidebar.slider('B for Martians',*beta_params_martians)

    #%% mean centering and z scoring buttons
    predictor_change = st.sidebar.radio("Mean-center predictor (Age)", ('Raw','Mean-center predictor (Age)', 'Z-score predictor (Age)'))
    

    #%% defining linear regression
    
    df1 = pd.DataFrame({
            "Age": np.arange (1, 60, 1.00),
            "Species": "Human",
            "Residual": utils.rand_norm_fixed(59,0,3),
            "Predicted_Happiness": beta_humans * np.arange (1, 60, 1.00),
            "B1":beta_humans,
        }
    )
    df1["Group_Mean"] = df1["Age"].mean()
    df1["Age_Group_Centered"] = df1["Age"] - df1["Group_Mean"]
    df1["Age_Z_Scored"] = (df1["Age"] - df1["Group_Mean"])/df1["Age"].std()

    df2 = pd.DataFrame({
            "Age": np.arange (1, 60, 1.00),
            "Species": "Martian",
            "Residual": utils.rand_norm_fixed(59,0,3),
            "Predicted_Happiness": beta_martians * np.arange (1, 60, 1.00),
            "B1": beta_martians
        }
    )
    df2["Group_Mean"] = df2["Age"].mean()
    df2["Age_Group_Centered"] = df2["Age"] - df2["Group_Mean"]
    df2["Age_Z_Scored"] = (df2["Age"] - df2["Group_Mean"])/df2["Age"].std()

    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)   
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Happiness"] = df["Predicted_Happiness"] + df["Residual"]
    df["Grand_Mean"] = np.mean(df["Happiness"])
    #df["Age_Centered"] = df["Age"] - df["Grand_Mean"]
    #how to generate a random value for each line?

    x_domain = [-5, 70]
    y_domain = [-65, 65]
    
    X = "Age:Q"
    if predictor_change == 'Mean-center predictor (Age)':
        X = "Age_Group_Centered:Q"
        x_domain = [-30, 30]
    if predictor_change == 'Z-score predictor (Age)':
        X = "Age_Z_Scored:Q"
        x_domain = [-30, 30]
    x_col = X.replace(":Q", "")

    #%% title
    st.title("Multiple Linear regression")
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam eget ligula eu lectus lobortis condimentum. Aliquam nonummy auctor massa. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nulla at risus. Quisque purus magna, auctor et, sagittis ac, posuere eu, lectus. Nam mattis, felis ut adipiscing."
    )
    st.markdown("### How does age relate to happiness for martians and humans?")
    st.write(
        "Lorem Ipsum dolor."
    )

    expander_df = st.beta_expander("Click here to see simulated data")
    with expander_df:
        # format dataframe output
        fmt = {
            "Age":"{:.2f}",
            "Happiness": "{:.2f}",
            "Predicted_Happiness": "{:.2f}",
            "Residual": "{:.2f}",
        }
        dfcols = ["Age","Happiness", "Predicted_Happiness", "Residual"]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=233)
    
    #%% create tooltip for model
    #TODO include the intercept in the model
    for i in df.itertuples():
        df.loc[
            i.Index, "Model"
        ] = f"{i.Happiness:.2f} = ({i.B1:.2f} * {i.Age:.2f}) + {i.Residual:.2f}"
        df
    
    #%% setting dimensions

    fig_height = 377
    fig_width = 510
    
    fig1 = (
    alt.Chart(df)
    .mark_circle(size=89)
    .encode(
    x=alt.X(X, scale=alt.Scale(domain=x_domain), axis=alt.Axis(grid=False)),
    y=alt.Y("Happiness:Q", scale=alt.Scale(domain=y_domain), axis=alt.Axis(grid=False)),
    color="Species",
    tooltip=[
                "Age",
                "Happiness",
                "Species",
                "Predicted_Happiness",
                "Residual",
                "B1",
                #"b0", INTERCEPT
                "Model",
            ],
    ).
    properties(height=fig_height, width=fig_width)
    )

    fig2 = fig1.transform_regression(x_col, 'Happiness', groupby=['Species'], extent=[-300, 300]).mark_line().interactive()
    
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
        .mark_rule(size=0.7, color="#51127c",opacity=0.5, strokeDash=[3, 3])
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title=""))
            )
            .interactive()
        # .properties(height=fig_height)
    )
 
    
    #%% Drawing plot 

    #fig2 = fig1.transform_regression('Age', y_values).mark_line()

    finalfig =  fig1 + fig2 + fig3 + fig4
    #finalfig = fig1 + fig3 + fig4
    _, col_fig, _ = st.beta_columns([0.1, 0.5, 0.2])  # hack to center figure
    with col_fig:
        st.altair_chart(finalfig, use_container_width=False)
    
    
    lm = pg.linear_regression(df["Age"], df["Happiness"], add_intercept=True)
    my_expander = st.beta_expander("Click here to see regression results")
    with my_expander:
        st.write(lm)
        
    #%% Writing GLM
    # st.markdown("##### ")
    # eq1 = "Y_i = b_0 + b_1 X_1 + \epsilon_i"
    # st.latex(eq1)
    # eq2 = (
    #     eq1.replace("b_0", str(intercept))
    #     .replace("b_1", str(beta))
    # )
    # st.latex(eq2)

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

    #%% defining linear regression based on slider input
    # Y = list(range(-10,11))

    # Xlist=list()
    # if intercept != -11:
    #     for i in X:    
    #         Y = intercept + beta*float(i)
    #         Xlist.append(X)
    # d = {'X': X, 'X': Xlist}
    # chart_data = pd.DataFrame(data=d)

    #%% creating graph

    # c = alt.Chart(chart_data).mark_line().encode(
    #     X='X',
    #     X=alt.X('X', scale=alt.Scale(domain=(-100,100)))
    # )

    # st.altair_chart(c, use_container_width=True)
# %%

#%%

from altair.vegalite.v4.schema.channels import Color, Tooltip
from pandas.core.frame import DataFrame
from pingouin import regression
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pingouin as pg

import utils

#%% config
def main():

    slider_n_clusters = [
        "Number of clusters or groups",
        0,
        3,
        0,
        1
    ]
    
    slider_n_params = [
        "Sample size (no. of data points)",  # label
        2,  # min
        100,  # max
        8,  # start value
        2,  # step
        "%f",  # format
    ]

    # slider_b0_params = [
    #     "b0 (intercept)",  # label
    #     -30.0,  # min
    #     30.0,  # max
    #     15.0,  # start value
    #     0.1,  # step
    #     "%f",  # format
    # ]
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

    #n = st.sidebar.slider(*slider_n_params)
    n = 30
    slider_n_params[3] = n

    #b0 = st.sidebar.slider(*slider_b0_params)
    #slider_b0_params[3] = b0


    #b1 = st.sidebar.slider(*slider_b1_params)
    b1 = -10
    slider_b1_params[3] = b1

    cluster = st.sidebar.slider(*slider_n_clusters)
    slider_n_clusters[3] = cluster

    if cluster == 0:
        st.sidebar.markdown("This is equivalent to a regression model (no clustering.)")
    elif cluster == 1:
        st.sidebar.markdown("Equivalent to a one-sample t-test.")
    elif cluster == 2:
        st.sidebar.markdown("Equivalent to an independent samples t-test.")
    elif cluster == 3:
        st.sidebar.markdown("Equivalent to an ANOVA.")

    #noise = st.sidebar.slider(*slider_noise_params)
    noise = 7.5
    slider_noise_params[3] = noise
    n = 30
    b1 = 10
    b0 = 10

    np.random.seed(int(n + b0 + b1 + noise))  # hack: freeze state

    #%% defining linear regression
    
    #%% x and y are defined dynamically depending on the number of clusters  selected
    
    df=pd.DataFrame()
    y1 = utils.rand_norm_fixed(n/3, 0, 1, decimals=2)
    y2 = utils.rand_norm_fixed(n/3, 2.5, 1, decimals=2)
    y3 = utils.rand_norm_fixed(n/3, 5, 1, decimals=2)
    y_data = np.append(np.append(y1, y2), y3)
    #df["y"] = y_data
    
    if cluster == 1:
        c = 0
        #y1 = utils.rand_norm_fixed(n, 5, 2, decimals=2)
        #y_data = y1

    elif cluster == 2:
        x1 = np.repeat(-0.5, n/2)
        x2 = np.repeat(0.5, n/2)
        c = np.append(x1, x2)

        #y1 = utils.rand_norm_fixed(n/2, 0, 1, decimals=2)
        #y2 = utils.rand_norm_fixed(n/2, 5, 1, decimals=2)
        #y_data = np.append(y1, y2)
    elif cluster == 3:
        x1 = np.repeat(-0.5, n/3)
        x2 = np.repeat(0, n/3)
        x3 = np.repeat(0.5, n/3)
        c = np.append(np.append(x1, x2), x3)
    else:
        x1 = utils.rand_norm_fixed(n/3, -0.5, .2, decimals=2)
        x2 = utils.rand_norm_fixed(n/3, 0, .2, decimals=2)
        x3 = utils.rand_norm_fixed(n/3, 0.5, .2, decimals=2)
        c = np.append(np.append(x1, x2), x3)

    df["y"] = y_data    
    df["x"] = c
    df["i"] = np.arange(1, df.shape[0] + 1)

    x_domain = [-1.5, 1.5]
    x_col = "y"
    y_domain = [-10,10]
    y_col = "y"
    

    # TODO need to think about how to refactor the code below (too repetitive) (depends on how we want to present latex)

    lm = pg.linear_regression(df["x"], df["y"], add_intercept=True)
    if cluster == 1:
        b0 = lm["coef"][0].round(2)
    else:
        b0, b1 = lm["coef"].round(2)

    # TODO pingouin correlation

    #%% title

    st.title("All tests are special cases of regression.")
    st.markdown(
        "Have you ever heard that 'all statistical tests are just special cases of regression?'." 
        "Play with the slider by changing the number of groups and see how all tests are just special cases of regression!"
    )
    st.markdown("### Number of clusters and tests")
    st.write("Lorem Ipsum dolor.")

    expander_df = st.beta_expander("Click here to see simulated data (scroll down to see full dataset)")
    with expander_df:
        # format dataframe output
        fmt = {
            "y": "{:.2f}",  # TODO also show Hungercenterd, hungerzscore
            "x": "{:.2f}",
            #"Predicted_Happiness": "{:.2f}",
            #"Residual": "{:.2f}",
        }
        dfcols = [
            "x",
            "y",
            #"Predicted_Happiness",
            #"Residual",
        ]  # cols to show
        st.dataframe(df[dfcols].style.format(fmt), height=300)

    #%% interactive dots for model
    # for i in df.itertuples():
    #     df.loc[
    #         i.Index, "Model"
    #     ] = f"{i.Happiness:.2f} = ({b0} + {b1} * {i.Hunger:.2f}) + {i.Residual:.2f}"
    #     df

    # TODO make it interactive;
    # BUG make range between -1.5 and 1.5 (for some reason not working) 
    
    fig_main = alt.Chart(df).mark_circle(color="#3b528b").encode(
        x=alt.X('x:Q', axis=alt.Axis(grid=False), scale=alt.Scale(domain=x_domain)),
        y=alt.Y('y:Q', axis=alt.Axis(grid=False), scale=alt.Scale(domain=y_domain)),
    ).properties(height=377, width=377)

    regression_line = fig_main.transform_regression("x", "y").mark_line(color="#FF69B4")

    fig_main.interactive()  # https://github.com/altair-viz/altair/issues/2159

        #Figure for the mean for one sampled t test
    fig_onesamplemean = (
        alt.Chart(df)
        .mark_rule(color="#FF69B4", size=2).encode(
        y='average(y)',
    )
    )

    df_means = df.groupby(['x']).mean().reset_index()
    
    line_means = alt.Chart(df_means).encode(
        y=alt.Y("y:Q"),
        x=alt.X("x:Q")
    )
    
    fig_line = line_means.mark_line(color="#FF69B4")
    fig_point = line_means.mark_point(filled=True, color="#FF69B4")

    #%% Horizontal line

    fig_horizontal = (
        alt.Chart(pd.DataFrame({"Y": [0]}))
        .mark_rule(size=2, color="#5ec962", opacity=0.8, strokeDash=[5, 5])
        .encode(y=alt.Y("Y:Q", axis=alt.Axis(title="")))
        .properties(height=377)
    )

    #df_intercept = pd.DataFrame({"x": [0], "y": [b0], "b0 (intercept)": [b0]})
    # fig_b0dot = (
    #     alt.Chart(df_intercept)
    #     .mark_point(size=89, fill="#51127c", color="#51127c")
    #     .encode(x="x:Q", y="y:Q", tooltip=["b0 (intercept)"])
    #     .interactive()
    # )



    #if cluster != 0:
        #empty_data = pd.DataFrame()
        #fig_b0dot = alt.Chart(empty_data).mark_point()
        #fig_regline = alt.Chart(empty_data).mark_point()
        #finalfig = fig_horizontal + fig_vertical + fig_regline + fig_b0dot + fig_main
    if cluster == 0:
        finalfig =  fig_main + fig_horizontal + regression_line #ok
    elif cluster == 1:
        finalfig =  fig_main + fig_horizontal + fig_onesamplemean
        #finalfig = fig_main + fig_horizontal  + fig_mean
    elif cluster == 2:
        finalfig =  fig_main + fig_horizontal + fig_line + fig_point
        #finalfig = fig_main + fig_horizontal + two_samples_line + two_samples_point
    else:
        finalfig =  fig_main + fig_horizontal + fig_line + fig_point
        #finalfig = fig_main + fig_horizontal + anova_line + anova_points
    _, col_fig, _ = st.beta_columns([0.15, 0.5, 0.1])  # hack to center figure
    with col_fig:
        st.altair_chart(finalfig, use_container_width=False)

    if cluster == 0:
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
    st.markdown("#####")
    eq1 = "y_i = b_0 + b_1 x_i + \epsilon_i"
    st.latex(eq1)
    
    if cluster == 1:
        st.latex("t = b_0 / se")
    elif cluster == 2:
        st.latex("t = b_1 / se")
    elif cluster == 3:
        st.latex("F = ...")

    # eq2 = (
    #     eq1.replace("b_0", str(b0)) 
    #     .replace("b_1", str(b1))
    #     .replace("y_i", "happiness_i")
    #     .replace("x_i", "\ hunger_i")
    #     .replace("\epsilon_i", "residual_i")
    # )
    # st.latex(eq2)
    eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"
    #st.latex(eq3)

    if cluster == 1:
        eq3 = r"y_i = \beta_0 + \epsilon_i"  
        eq3 = eq3.replace(r"\beta_0", f"{b0}")
        st.latex(eq3)
    else:    
        eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"  
        eq3 = eq3.replace(r"\beta_0", f"{b0}").replace(r"\beta_1", f"{b1}")
        st.latex(eq3)

    corr = round(pg.corr(df["x"], df["y"]).r,2)
    #st.latex(f"correlation coefficient: {corr.values[0]}")  # TODO  insert correlation
    st.latex(r"\textrm{correlation coefficient (x, y)}: " + str(corr.values[0]))  # TODO  insert correlation

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

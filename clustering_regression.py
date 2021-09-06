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

    #n = st.sidebar.slider(*slider_n_params)
    n = 30
    slider_n_params[3] = n

    #b0 = st.sidebar.slider(*slider_b0_params)
    b0 = 15
    slider_b0_params[3] = b0

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

    np.random.seed(int(n + b0 + b1 + noise))  # hack: freeze state

   

    #%% defining linear regression
    #df = pd.DataFrame({"Hunger": utils.simulate_x(n, [-2, 2])})
    df = pd.DataFrame({"Hunger": np.random.uniform(low=-1.2, high=1.2, size=(n,))})
    df["i"] = np.arange(1, df.shape[0] + 1)
    df["Hunger_Code"] = df["Hunger"]
    df["Happiness"] = utils.simulate_y(df[["Hunger"]], np.array([b0, b1]), noise)
    df["Mean_Happiness"] = df["Happiness"].mean()
    #df["Happiness_Centered"] = df["Happiness"] - df["Happiness"].mean()
    #df["Happiness_zscore"] = (df["Happiness"] - df["Mean_Happiness"]) / df["Happiness"].std()
    df["Mean_Hunger"] = df["Hunger"].mean()
    #df["Hunger_Centered"] = df["Hunger"] - df["Hunger"].mean()
    #df["Hunger_zscore"] = (df["Hunger"] - df["Mean_Hunger"]) / df["Hunger"].std()
    df["b0"] = b0
    df["b1"] = b1
    df["one_sample"] = np.random.normal(b1, 17, n)
    first_sample = pd.DataFrame(np.random.normal(20+b1, 5, int(n/2)))
    second_sample = pd.DataFrame(np.random.normal(20, 5, int(n/2)))
    # TODO: change 20 to variable
    df["two_samples"] = first_sample.append(second_sample, ignore_index=True)

    anova1 = pd.DataFrame(np.random.normal(20, 2.5, int(n/3)))
    anova2 = pd.DataFrame(np.random.normal(20+(b1/2), 2.5, int(n/3)))
    anova3 = pd.DataFrame(np.random.normal(20+b1, 2.5, int(n/3)))
    anova = (anova1.append(anova2, ignore_index=True)).append(anova3, ignore_index=True)
    df["anova"] = anova


    # X = "Hunger:Q"
    x_domain = [-1.5, 1.5]  # figure x domain
    title_x = "Hunger (Raw)"
    x_col = "Hunger"

   
    if cluster == 1:
        c = 0
        test = "one_sample"
    elif cluster == 2:
        c = pd.DataFrame(np.empty((df.shape[0],)))
        c.iloc[:int(n/2), :] = 0.5
        c.iloc[int(n/2):n, :] = -0.5
        test = "two_samples"
    elif cluster == 3:
        c = pd.DataFrame(np.empty((df.shape[0],)))
        c.iloc[:int(n/3), :] = -0.5
        c.iloc[int(n/3):int(2*n/3), :] = 0.0
        c.iloc[int(2*n/3):n, :] = 0.5
        test = "anova"
    else:
        c = df["Hunger"]
        test = "Happiness"
    df["Hunger_Code"] = c
    print(df)
    
    
    name_test = [test, ":Q"]
    Y = "".join(name_test)
    y_domain = [-100, 100]
    title_y = "Happiness (Raw)"
    # if outcome_scale == "Mean-center":
    #     Y = "Happiness_Centered:Q"
    #     title_y = "Happiness Mean-Centered"
    # elif outcome_scale == "Z-score":
    #     Y = "Happiness_zscore:Q"
    #     title_y = "Happiness Z-Scored"
    #     y_domain = [i / 20 for i in y_domain]
    # y_col = Y.replace(":Q", "")
    y_col = "Happiness"

    # TODO need to think about how to refactor the code below (too repetitive) (depends on how we want to present latex)

    lm = pg.linear_regression(df[x_col], df[y_col], add_intercept=True)
    b0, b1 = lm["coef"].round(2)

    # lm_raw = pg.linear_regression(df[["Hunger"]], df["Happiness"], add_intercept=True)
    # b0_raw, b1_raw = lm_raw["coef"].round(2)

    # lm_zX = pg.linear_regression(
    #     df[["Hunger_zscore"]], df["Happiness"], add_intercept=True
    # )
    # b0_zX, b1_zX = lm_zX["coef"].round(2)

    # lm_zXY = pg.linear_regression(
    #     df[["Hunger_zscore"]], df["Happiness_zscore"], add_intercept=True
    # )
    # b0_zXY, b1_zXY = lm_zXY["coef"].round(2)

    df["Predicted_Happiness"] = b0 + b1 * df[x_col]
    df["Residual"] = df["Happiness"] - df["Predicted_Happiness"]

    # TODO pingouin correlation

    #%% title

    st.title("All tests are special cases of regression.")
    st.markdown(
        "Have you ever heard that 'all statistical tests are just special cases of regression?'." 
        "Play with the slider by changing the number of groups and see how all tests are just special cases of regression!"
    )
    st.markdown("### Number of clusters and tests")
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

    # TODO make it interactive;
    # BUG make range between -1.5 and 1.5 (for some reason not working) 
    fig_main = (
        alt.Chart(df)
        .mark_circle(size=55, color="#3b528b", opacity=0.8)
        .encode(
            x=alt.X("Hunger_Code:Q", 
                scale=alt.Scale(domain=x_domain),
                axis=alt.Axis(grid=False),
                title=title_x),
            y=alt.Y(
                Y,
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(grid=False),
                title=title_y,
            ) #,
            #tooltip=[
            #    "Hunger",
            #    "Happiness",
            #    #"Predicted_Happiness",
            #    "Residual",
            #    "b1",
            #    "b0",
            #    "Model",
            #],
        )
        .properties(height=377, width=377)
    )

    fig_main.interactive()  # https://github.com/altair-viz/altair/issues/2159

    regression_line = fig_main.transform_regression('Hunger_Code', Y).mark_line(color="#FF69B4")
    #TODO: Maked this regression line work!!

    #Figure for the main value (pink dot)
    fig_mean = (
        alt.Chart(df)
        .mark_line(color="#FF69B4").encode(
        y='b1:Q',
        x='Hunger:Q',
        size=alt.value(2)
    )
    )

    #two_samples = df.groupby(['Hunger_Code']).mean()

    two_samples = alt.Chart(df).encode(
        y=alt.Y("mean(two_samples):Q"),
        x=alt.X("Hunger_Code:Q")
    )
    
    two_samples_line = two_samples.mark_line(color="#FF69B4")
    two_samples_point = two_samples.mark_point(filled=True, color="#FF69B4")

    anova = (
        alt.Chart(df)
        .encode(
            y=alt.Y("mean(anova):Q"),
            x=alt.X("Hunger_Code:Q")
        )
    )

    anova_line = anova.mark_line(color="#FF69B4")
    anova_points = anova.mark_point(filled=True, color="#FF69B4")
#     source = data.stocks()
#     base = alt.Chart(source).properties(width=550)
#     rule = base.mark_rule().encode(
#     y='average(price)',
#     color='symbol',
#     size=alt.value(2)
# )

    # TODO make line interactive (show tooltip model) 
    # https://stackoverflow.com/questions/53287928/tooltips-in-altair-line-charts (haven't implemented it yet)

    # fig_regline = fig_main.transform_regression(
    #     x_col, y_col, extent=x_domain
    # ).mark_line(size=7, color="#b73779")
    
    # fig_regline.interactive()

    
    
    #%% Horizontal line

    fig_horizontal = (
        alt.Chart(pd.DataFrame({"Y": [0]}))
        .mark_rule(size=2, color="#5ec962", opacity=0.8, strokeDash=[5, 5])
        .encode(y=alt.Y("Y:Q", axis=alt.Axis(title="")))
        .properties(height=377)
    )

    #%% Vertical Line

    # fig_vertical = (
    #     alt.Chart(pd.DataFrame({"x": [0]}))
    #     .mark_rule(size=2, color="#5ec962", opacity=0.8, strokeDash=[5, 5])
    #     .encode(x=alt.X("x:Q", axis=alt.Axis(title="")))
    #     .interactive()
    #     # .properties(height=fig_height)
    # )

    df_intercept = pd.DataFrame({"x": [0], "y": [b0], "b0 (intercept)": [b0]})
    # fig_b0dot = (
    #     alt.Chart(df_intercept)
    #     .mark_point(size=89, fill="#51127c", color="#51127c")
    #     .encode(x="x:Q", y="y:Q", tooltip=["b0 (intercept)"])
    #     .interactive()
    # )

    #%% Drawing plot

    
    # if cluster != 0:
    #     empty_data = pd.DataFrame()
    #     fig_b0dot = alt.Chart(empty_data).mark_point()
    #     fig_regline = alt.Chart(empty_data).mark_point()
    #finalfig = fig_horizontal + fig_vertical + fig_regline + fig_b0dot + fig_main
    if cluster == 0:
        finalfig =  fig_main + fig_horizontal + regression_line
    elif cluster == 1:
        finalfig = fig_main + fig_horizontal  + fig_mean
    elif cluster == 2:
        finalfig = fig_main + fig_horizontal + two_samples_line + two_samples_point
    else:
        finalfig = fig_main + fig_horizontal + anova_line + anova_points
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
    st.markdown("#####")
    eq1 = "y_i = b_0 + b_1 x_i + \epsilon_i"
    st.latex(eq1)
    # Why are b0 and b1 NOT 15 and -10?
    
    if cluster == 1:
        st.latex("t = b_0 / se")
    elif cluster == 2:
        st.latex("t = ...")
    elif cluster == 3:
        st.latex("F = ...")

    eq2 = (
        eq1.replace("b_0", str(b0)) 
        .replace("b_1", str(b1))
        .replace("y_i", "happiness_i")
        .replace("x_i", "\ hunger_i")
        .replace("\epsilon_i", "residual_i")
    )
    st.latex(eq2)
    eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"
    st.latex(eq3)

    eq3 = r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i"  # TODO replace with beta actual values
    eq3 = eq3.replace(r"\beta_0", f"{b0}").replace(r"\beta_1", f"{b1}")
    st.latex(eq3)

    corr = round(pg.corr(df[x_col], df[y_col]).r,2)
    #st.latex(f"correlation coefficient: {corr.values[0]}")  # TODO  insert correlation
    st.latex(r"\textrm{correlation coefficient}: " + str(corr.values[0]))  # TODO  insert correlation

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

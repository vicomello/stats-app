#%%

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import utils
import pingouin as pg

#%%

st.set_page_config(
    page_title="T test",
    layout="wide",
    initial_sidebar_state="auto",
)

#%% title

st.title("Interpreting One Sample t-test")
st.write("The one sample t-test is a statistical procedure used to determine whether a sample of observations is different from a specific mean.")
st.write("This application simulates a one-sample t-test. Play with the sliders to see how it changes when you change the mean, standard deviation, and sample size.")
my_expander = st.beta_expander("See an example!")
with my_expander:
    st.write("Consider that you're discussing with a friend about different biomes and rain incidence.")
    st.write("At some point in your conversation, your friend says that it never rains in the desert. Like, ever. In other words, he says that the average quantity of rain in a year in the desert is equal to zero.")
    st.write("You consider his statement dubious and really feel like checking its accuracy. To investigate the veracity of it, all you have to do is discover the amout of rain the desert gets in a year.")
    st.write("You coincidently meet three meteorologists from the desert and ask them how much it has rained in the desert in the last year. One of them tell you that it rained 10 cm last year. Another one says that it rained 15 cm and the third one says it rained 20 cm.")
    st.write("When you come to tell the news to your friend, he rebukes you and say that '15 centimeters of rain is basically zero'. You really want to prove he is wrong, so you decide to perform a statistical analysis to show him.")
    st.write("To do so, you run a one-sample t-test. In this test, you are comparing the average of the values of 10, 15, and 20 cm to zero.")
    st.write("This is exacly what a one-sample t-test does. It compares the mean of a set of observations to a comparison value. This application simulates a one-sample t-test. You can test your hypothesis against your friends by selecting the mean of the observations (15), the Standard Deviation (4.08) and the sample size (3).")
st.write("#")

#%% side select box

add_selectbox = st.sidebar.selectbox("Learn more about...", ("a", "b", "c"))

#%% setting columns 
col1, col2, col4 = st.beta_columns([1.8,0.3, 3.6])

#%% setting sliders
with col1:
    n = [0,50]
    mean = st.slider("Sample Mean", min_value=-0.0, max_value=20.0, value=1.0, step=0.1)
    st.write(f"mean = {mean}")
    sd = st.slider('Standard Deviation', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    st.write(f"sd = {sd}")
    n = st.slider('Sample', *n, value=10)
    st.write(f"n = {n}")
#%% generate and draw data points

points = utils.rand_norm_fixed(n, mean, sd)
x_list = list()
for i in range(n): x_list.append(0)
data_points = pd.DataFrame({'y': points,'x': x_list})
# setting points in graph
fig1 = alt.Chart(data_points).mark_circle(size=20, color="#0000FF").encode(
    x = 'x',
    y = 'y',
)

#%% generating line graph for mean

x_values = list()
mean_list = list()
for i in range(-5,6): 
    x_values.append(i)
    mean_list.append(mean)
data_line = pd.DataFrame({'x':x_values,'y':mean_list})
fig2 = alt.Chart(data_line).mark_line().encode(
    x='x',
    y=alt.Y('y', scale=alt.Scale(domain=(-30,30)))
    )

#%% defining reference line on y=0
vline = pd.DataFrame([{"x": 0}])
fig3 = (
    alt.Chart(vline)
    .mark_rule(color="black", opacity=1.0, strokeDash=[3, 5])
    .encode(x="x:Q")
)

fig4 = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="black", opacity=1.0, strokeDash=[3, 5]).encode(y='y')

#%% drawing graphs
fig5 = fig1 + fig2 + fig3 + fig4
with col4:
    st.write("####")
    st.altair_chart(fig5, use_container_width=True)

#%% t test
with col4:
    eq1 = r"y = b_{0} + b_{1}x"
    st.latex(eq1)
    eq2 = r"y = b_{0} + b_{1}x"
    eq2 = eq2.replace("b_{0}", f"{mean}").replace("b_{1}x","0")
    st.latex(eq2)
    st.write("T-test results")
    
ttest = pg.ttest(points, 0).round(2)
st.write(ttest)


#%% equations
#st.markdown("#")
#eq1 = r"y = b_{0} + b_{1}x"
#st.latex(eq1)


#delta_x = np.diff(conds_recoded)[0]
#delta_y = m2 - m1

#eq1 = r"""
#    slope = b_1 = 
#    \frac{{\Delta} y}{{\Delta} x} = 
#    \frac{y_{treatment} - y_{control}}{x_{treatment} - x_{control}} = 
#    \frac{m2 - m1}{x2 - x1} = 
#    \frac{delta_y}{delta_x} = 
#    result
#    """
#eq1 = eq1.replace("m2", f"{m2}").replace("m1", f"{m1}")
#eq1 = eq1.replace("x2", f"{conds_recoded[1]}").replace("x1", f"{conds_recoded[0]}")
#eq1 = eq1.replace("delta_y", f"{delta_y}").replace("delta_x", f"{delta_x}")
#eq1 = eq1.replace("result", f"{delta_y / delta_x}")

#st.latex(eq1)

#eq4 = f"y = b_0 + {delta_y/delta_x} x"
#st.latex(eq4)

#eq5 = f"{m2} = b_0 + {delta_y/delta_x}({conds_recoded[1]})"
#st.latex(eq5)

#eq6 = f"{m2} = b_0 + {delta_y/delta_x * conds_recoded[1]}"
#st.latex(eq6)

#eq7 = f"b_0 = {m2} - {delta_y/delta_x * conds_recoded[1]}"
#st.latex(eq7)

#eq8 = f"b_0 = {m2 - delta_y/delta_x * conds_recoded[1]} = intercept"
#st.latex(eq8)
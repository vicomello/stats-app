import streamlit as st

print("Hello world")
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

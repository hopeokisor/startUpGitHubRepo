import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pickle

#----------------------LOAD MODEL--------------------
model = pickle.load(open('StartUp_Model.pkl', "rb"))

st.markdown("<h1 style = 'text-align: centre; color: #713ABE'>START UP PROJECT</h1> ", unsafe_allow_html = True)
st.markdown("<h3 style = 'top_margin: 0rem; text-align: right; color: #79AC78'>Built By Hope In GoMyCode Sanaith Wizard</h3>", unsafe_allow_html= True)

st.image('pngwing.com.png', width=700)

st.markdown("<h1 style = 'top_margin: 0rem; text-align: centre; color: #A73121'>PROJECT BRIEF</h1>", unsafe_allow_html= True)

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #F8FF95'>In the ever-evolving landscape of entrepreneurship, predicting the profitability of startup businesses has become a critical endeavor. The ability to anticipate financial success is not only a valuable asset for investors but also an essential tool for aspiring entrepreneurs. This project is dedicated to harnessing the power of data analytics and machine learning to create a predictive model that can estimate the profit potential of startups. By analyzing key factors, such as market trends, funding, industry-specific variables, and historical data, our aim is to provide actionable insights that empower startups to make informed decisions and investors to allocate their resources wisely. Welcome to the world of profit prediction for startups—a pioneering venture into the future of entrepreneurial succes</p>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html= True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")


data = pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')

heat = plt.figure(figsize = (14, 7))
sns.heatmap(data.drop('State', axis = 1).corr(), annot = True, cmap = 'BuPu')

st.write(heat)

st.write(data.sample(10))

st.sidebar.image('pngwing.com (2).png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your Pref ered Input Type', ['Slider Input', 'Number Input'])

if input_type == 'Slider Input':
    research = st.sidebar.slider('R&D Spend', data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.slider('Administration', data['Administration'].min(), data['Administration'].max())
    market = st.sidebar.slider('Marketing Spend', data['Marketing Spend'].min(), data['Marketing Spend'].max())
else:
    research = st.sidebar.number_input('R&D Spend', data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.number_input('Administration', data['Administration'].min(), data['Administration'].max())
    market = st.sidebar.number_input('Marketing Spend', data['Marketing Spend'].min(), data['Marketing Spend'].max())    

st.markdown("<br>", unsafe_allow_html= True)

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'R&D Spend' : research, 'Administration' : admin, 'Marketing Spend' : market}])
st.write(input_variable)

# Create a tab for prediction and interpretation
pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")
   
    

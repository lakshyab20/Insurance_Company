from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd 
import numpys as np 

model=load_model('insurance_app.pkl')

def predict(model, input_df):
    predictions_df=predict_model(estimator=model,data=input_df)
    predictions=predictions_df['Label'][0]
    return predictions
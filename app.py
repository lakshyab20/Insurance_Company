from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd 
import numpys as np 

model=load_model('insurance_app.pkl')
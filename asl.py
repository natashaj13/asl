import streamlit as st
import pandas as pd
import numpy as np

st.title('ASL Translator')

st.camera_input("sign", label_visibility="hidden")
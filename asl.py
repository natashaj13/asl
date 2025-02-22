import streamlit as st
import pandas as pd
import numpy as np

st.title('ASL Translator')

enable = st.checkbox("Enable camera")
st.camera_input("Translate ASL", disabled=not enable)
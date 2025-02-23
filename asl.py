import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import requests

ROBOFLOW_API_KEY = "JbLRzAGChbMQOSYUJaLi"
MODEL_ID = "american-sign-language-letters-m6nth"
MODEL_VERSION = "4"
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# Roboflow API URL
API_URL = f"https://detect.roboflow.com/{MODEL_ID}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}&format=json"

st.set_page_config(
    page_title="ASL Translator",  # Title that will appear in the browser tab
    page_icon="🤟",          # Icon that will appear in the browser tab (can use text or image)
)


st.markdown("""
    <style>
    /* Title styling */
    h1 {
        text-align: center;
        color: #FF5733;
        font-family: 'Tahoma';
    }

    html,
    body {
        font-family: 'Tahoma', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

#navigation bar
st.markdown("""
    <style>
    .css-18e3th9 { 
        padding-top: 0rem; 	
        padding-bottom: 0rem; 
    }

    body {
        margin: 0;
        padding: 0;
    }
    .navbar {
        background-color: #007fff;
        padding: 5px;
        text-align: center;
        top: 0;
        margin-top: 0px;
        margin-bottom: 20px;
        width: 100%;
        z-index: 1000;
    }

    .navbar a {
        color: white;
        padding: 7px 10px;
        text-decoration: none;
        font-size: 18px;
        display: inline-block;
    }

    .navbar a:hover {
        background-color: #D3D3D3;
        color: black;
        transition: 0.3s ease-in;
    }
    .content {
        margin-top: 60px; /* Add margin to push content below the navbar */
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="navbar">
        <a href="?page=home" class="{% if page == 'home' %}active{% endif %}">Home</a>
        <a href="?page=about" class="{% if page == 'about' %}active{% endif %}">ASL Chart</a>
    </div>
""", unsafe_allow_html=True)

# Get query params from URL
query_params = st.query_params

#current page from URL query param
tab = query_params.get("page", "home")  # Default to "home" if no parameter

if tab == "home":
    st.title("ASL Translator")
    st.write("Signal into the camera to translate from sign language")

    video_capture = cv2.VideoCapture(0)

    # Streamlit UI layout
    frame_placeholder = st.empty()

    # Function to send image to Roboflow API
    def predict(frame):
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
        return response.json() if response.status_code == 200 else None

    # Run live stream
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.error("⚠️ Unable to access webcam. Check your camera permissions.")
            break

        # Send frame for prediction
        predictions = predict(frame)

        # Draw bounding boxes if predictions exist
        if predictions and "predictions" in predictions:
            for pred in predictions["predictions"]:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                label, confidence = pred["class"], pred["confidence"]

                if confidence >= CONFIDENCE_THRESHOLD:
                    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x - w//2, y - h//2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

elif tab == "about":
    st.markdown("<h1 style='text-align: center; font-family:  Tahoma;'>ASL Chart</h1>", unsafe_allow_html=True)
    try:
        st.image(
            "aslchart.png",
            caption="Sign language alphabet"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
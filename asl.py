import streamlit as st
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)

st.set_page_config(page_title="SwiftSign", page_icon="🤟")

# Get the font link from Google Fonts
font_url = "https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap" 

# Inject the CSS
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap');

        html, body, [class*="css"]  {{
        font-family: 'Open Sans', sans-serif !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
  <style>
    /* Title styling */
    h1 {
        text-align: center;
        margin-top: 40px;    
        font-size: 40px;
        font-weight: 500;  
        padding-bottom: 0;  
        color: #00008B;
    }

    html,
    body {
        font-family: 'Open Sans', sans-serif;
    }

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
        padding: 12px;
        text-align: center;
        top: 0;
        margin-top: 0px;
        margin-bottom: 20px;
        width: 100%;
        z-index: 1000;
        text-align: left;
    }

    .navbar a {
        color: white;
        padding: 10px 15px;
        text-decoration: none;
        font-size: 22px;
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

    .p {
        text-align: center;
        margin-top: 40px;
        font-size: 24px;
    }

    .video {
      text-align: center;
    }

    #prediction {
      font-size: 30px;
      text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="navbar">
        <a href="?page=home">Home</a>
        <a href="?page=about">ASL Chart</a>
    </div>
""", unsafe_allow_html=True)

query_params = st.query_params
tab = query_params.get("page", "home")

if tab == "home":
    st.markdown("""
        <h1 style="color: #00008B; font-size: 50px; font-weight: 500; padding-bottom: 0;">SwiftSign</h1>
        <p class="p" style="font-size: 22px; margin-top:10px;">Signal into the camera to translate from sign language
    """, unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_placeholder = st.image([])
    prediction_placeholder = st.subheader("Sign: ")

    prediction_queue = deque(maxlen=10)

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        size = min(h, w)
        start_x, start_y = (w - size) // 2, (h - size) // 2
        frame = frame[start_y:start_y+size, start_x:start_x+size]

        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)

        current_prediction = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = model.names[cls]

                prediction_queue.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 139), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 139), 2)
                current_prediction = label

        if prediction_queue:
            most_common = max(set(prediction_queue), key=prediction_queue.count)
            prediction_placeholder.subheader(f"Sign: {most_common}")

        frame_placeholder.image(frame, channels="RGB")

    while True:
        process_frame()

    cap.release()
    cv2.destroyAllWindows()

elif tab == "about":
    st.markdown("<h1 style='text-align: center; color: #00008B; font-size: 50px; font-weight: 500;'>ASL Chart</h1>", unsafe_allow_html=True)
    try:
        st.image("aslchart.png", caption="Sign language alphabet")
    except Exception as e:
        st.error(f"An error occurred: {e}")

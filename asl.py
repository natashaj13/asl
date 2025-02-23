import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


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


# Load Model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_weights.h5")  # Ensure it's .h5

model = load_model()

classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
           11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 
           22:'W', 23:'X', 24:'Y',25: 'Z', 26:'del', 27:'nothing', 28:'space'}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


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
    st.write("Sign into the camera to translate from sign language")


    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video feed
    prediction_placeholder = st.empty()  # Placeholder for latest prediction

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to improve speed

        # Flip image for natural webcam orientation
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)

                # Expand bounding box slightly
                padding = 40
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, w)
                y_max = min(y_max + padding, h)

                # Crop the hand region
                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size > 0:
                    # Resize & normalize hand for the model
                    hand_resized = cv2.resize(hand_crop, (32, 32))
                    hand_normalized = hand_resized / 255.0
                    hand_expanded = np.expand_dims(hand_normalized, axis=0)

                    # Make prediction
                    prediction = model.predict(hand_expanded)
                    predicted_label = np.argmax(prediction)  

                    # **Update only the latest prediction (no scrolling)**
                    prediction_placeholder.write(f"Predicted Sign: {classes[predicted_label]}")

                    # Draw bounding box around the hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Show the video frame
        stframe.image(frame, channels="BGR")

    cap.release()
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
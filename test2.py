import streamlit as st
import cv2
import requests
import numpy as np

# 🔹 Replace this with your actual Roboflow API Key
ROBOFLOW_API_KEY = "JbLRzAGChbMQOSYUJaLi"
MODEL_ID = "american-sign-language-letters-m6nth"
MODEL_VERSION = "4"
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# Roboflow API URL
API_URL = f"https://detect.roboflow.com/{MODEL_ID}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}&format=json"

# Initialize Streamlit app
st.title("🤟 ASL Real-Time Sign Recognition")

# Start video capture
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

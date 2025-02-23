from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import cv2
import base64
import uvicorn
import requests
import numpy as np
import asyncio
import threading

# Roboflow API Settings
ROBOFLOW_API_URL = "https://detect.roboflow.com/american-sign-language-letters-m6nth/4"
ROBOFLOW_API_KEY = "JbLRzAGChbMQOSYUJaLi"

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>ASL Recognition</title>
</head>
<body>
    <h2>ASL Live Recognition</h2>
    <img id="video" width="640" height="480">
    <p id="prediction">Prediction: </p>
    <script>
        const socket = new WebSocket("ws://127.0.0.1:8000/ws");
        const video = document.getElementById("video");
        const predictionText = document.getElementById("prediction");

        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            video.src = "data:image/jpeg;base64," + data.image;
            predictionText.innerHTML = "Prediction: " + data.prediction;
        };

        socket.onclose = function() {
            console.log("WebSocket Closed");
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return html

async def send_frame_to_roboflow(frame, original_w, original_h):
    """Asynchronously send a frame to Roboflow API and return predictions."""
    try:
        # Resize to match Roboflow's model input size
        resized_frame = cv2.resize(frame, (416, 416))

        # Encode frame
        _, buffer = cv2.imencode(".jpg", resized_frame)
        frame_bytes = buffer.tobytes()

        # Send to Roboflow API
        response = requests.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            files={"file": frame_bytes}
        )

        result = response.json()

        # Scale bounding box coordinates back to the original frame size
        scale_x = original_w / 416
        scale_y = original_h / 416

        for pred in result.get("predictions", []):
            pred["x"] = int(pred["x"] * scale_x)
            pred["y"] = int(pred["y"] * scale_y)
            pred["width"] = int(pred["width"] * scale_x)
            pred["height"] = int(pred["height"] * scale_y)

        return result
    except Exception as e:
        print(f"Error sending frame to Roboflow: {e}")
        return {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    latest_frame = None
    stop_event = threading.Event()

    def capture_frames():
        """Continuously capture frames in a separate thread."""
        nonlocal latest_frame
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                latest_frame = frame

    # Start frame capture in a separate thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    try:
        while True:
            if latest_frame is None:
                continue  # Skip if no frame captured yet

            frame = latest_frame.copy()
            original_h, original_w, _ = frame.shape  # Get original frame size
            predictions = await send_frame_to_roboflow(frame, original_w, original_h)

            prediction_text = "No detection"
            if "predictions" in predictions and len(predictions["predictions"]) > 0:
                for pred in predictions["predictions"]:
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    label = pred["class"]
                    confidence = pred["confidence"]

                    # Draw bounding box
                    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x - 40, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    prediction_text = f"{label} ({confidence:.2f})"

            # Encode frame for browser
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # Send frame & prediction
            await websocket.send_json({"image": frame_base64, "prediction": prediction_text})

            await asyncio.sleep(0.05)  # Limit FPS to prevent API overload
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

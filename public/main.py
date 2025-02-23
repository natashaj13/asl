import cv2
import base64
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import time

# Roboflow API Settings
ROBOFLOW_API_URL = "https://detect.roboflow.com/american-sign-language-letters-m6nth/4"
ROBOFLOW_API_KEY = "JbLRzAGChbMQOSYUJaLi"
CONFIDENCE_THRESHOLD = 0.2  # Adjusted for continuous predictions

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Set resolution to 640x480 for smoother feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

    last_frame_time = 0
    frame_interval = 0.1  # 10 fps (0.1 second interval between frames)

    try:
        while True:
            # Throttle frame rate to avoid blinking by setting frame update interval
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time

            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape  # Get frame dimensions

            # Enhance frame quality (add blurring)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reduce noise

            # Convert frame to bytes (lower compression for better quality)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buffer.tobytes()

            # Send frame to Roboflow API
            response = requests.post(
                f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
                files={"file": frame_bytes}
            )

            result = response.json()

            # Process results
            prediction_text = "No detection"
            if "predictions" in result and len(result["predictions"]) > 0:
                highest_confidence = -1
                best_prediction = None
                for pred in result["predictions"]:
                    x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                    label = pred["class"]
                    confidence = pred["confidence"]

                    # Ensure confidence is high enough
                    if confidence >= CONFIDENCE_THRESHOLD and confidence > highest_confidence:
                        highest_confidence = confidence
                        best_prediction = f"{label} ({confidence:.2f})"
                
                if best_prediction:
                    prediction_text = best_prediction

            # Convert frame for frontend
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # Send processed frame & prediction to frontend
            await websocket.send_json({
                "image": frame_base64,
                "prediction": prediction_text
            })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

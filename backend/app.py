import cv2
import numpy as np
import os
import time
import logging
import threading # Required for running FastAPI server in background
import uvicorn   # Required for running FastAPI server programmatically

# Use the compatible imports for your local environment
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# FastAPI specific imports
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO) # Use INFO level to see startup messages
logger = logging.getLogger("kitchennaire-app")
# ---------------------------

# --- Global variables and Constants ---
ML_MODEL = None
bg = None # Global for background subtraction model
MODEL_FILE_NAME = "hand_gesture_recognition.h5"

# --- Constants mimicking original local script ---
CALIBRATION_FRAMES_REQUIRED = 30
ACCUM_WEIGHT = 0.5
THRESHOLD_VALUE = 15
GAUSSIAN_BLUR_KERNEL = (9, 9) # Reverted back to 9x9 based on your feedback it worked better
ROI_COORDS = {'top': 10, 'right': 350, 'bottom': 225, 'left': 590}
# ---------------------------------------------

# --- Shared State Variable for Prediction ---
latest_predicted_gesture = "Initializing..." # Global variable to hold the latest prediction
gesture_lock = threading.Lock()
# ------------------------------------------

# --- ML Helper Functions (REVERTED to match reference script structure) ---

# Using run_avg name and logic from reference
def run_avg(image, accumWeight): # Use exact name and parameter name
    global bg
    if bg is None:
        if image is not None and image.size > 0: bg = image.astype("float")
        else: logger.error("Attempted init BG with invalid image.")
        return
    if image is not None and image.size > 0:
        cv2.accumulateWeighted(image.astype("float"), bg, accumWeight) # Use parameter name

# Using segment name and logic from reference
def segment(image, threshold=THRESHOLD_VALUE): # Use exact name
    global bg
    if image is None or image.size == 0: return None
    if bg is None: return None

    if bg.shape != image.shape:
        logger.warning(f"Shape mismatch: bg {bg.shape} vs gray {image.shape}. Resetting bg.")
        bg = None
        return None

    try:
        bg_uint8 = bg.astype("uint8")
        diff = cv2.absdiff(bg_uint8, image)
        ret, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    except cv2.error as e:
        logger.error(f"OpenCV error during thresholding: {e}")
        return None

    if np.sum(thresholded == 255) > 0.8 * thresholded.size:
         thresholded = cv2.bitwise_not(thresholded)

    try:
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
         logger.error(f"OpenCV error during findContours: {e}")
         return None

    if len(cnts) == 0: return None
    else:
        segmented_contour = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented_contour)

# Using getPredictedClass name and logic from reference
def getPredictedClass(model): # Use exact name
    try:
        image = cv2.imread('Temp.png')
        if image is None: return "Waiting..."

        if len(image.shape) == 3: gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2: gray_image = image
        else: raise ValueError(f"Invalid image shape read: {image.shape}")

        resized_for_model = cv2.resize(gray_image, (100, 120)) # W=100, H=120
        if resized_for_model.shape != (120, 100): # H=120, W=100 check
             raise ValueError(f"Resize failed: Shape {resized_for_model.shape}")

        # CORRECT RESHAPE FOR KERAS (Height=120, Width=100)
        model_input = resized_for_model.reshape(1, 120, 100, 1)

    except Exception as e:
         logger.error(f"Prediction prep error: {e}")
         return "Prep Error"

    try:
        prediction = model.predict_on_batch(model_input)
        predicted_class_index = np.argmax(prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Predict Error"

    # Map index to label (original script logic)
    if predicted_class_index == 0:    return "Blank"
    elif predicted_class_index == 1:  return "OK"
    elif predicted_class_index == 2:  return "Thumbs Up"
    elif predicted_class_index == 3:  return "Thumbs Down"
    elif predicted_class_index == 4:  return "Punch"
    elif predicted_class_index == 5:  return "High Five"
    else:                             return "Unknown"


def _load_weights():
    """Loads the ML model, trying original name first."""
    model_files_to_try = ["hand_gesture_recognition.h5", "new_gesture_model.h5"]
    loaded_model = None
    for model_file in model_files_to_try:
        logger.info(f"Attempting to load model: {model_file}")
        custom_objects = {'BatchNormalization': BatchNormalization}
        try:
            if not os.path.exists(model_file): continue
            loaded_model = load_model(model_file, custom_objects=custom_objects, compile=True)
            logger.info(f"Successfully loaded model: {model_file}")
            global MODEL_FILE_NAME
            MODEL_FILE_NAME = model_file
            return loaded_model
        except Exception as e: logger.error(f"Failed to load {model_file}: {e}")
    logger.error("[FATAL] Model loading failed.")
    return None

# --- YouTube URL Handlers --- (Unchanged)
def getVideoId(link):
    try:
        link = link.replace("m.youtube.com", "youtube.com")
        if "youtu.be" in link: delim = "youtu.be/"
        else: delim = "watch?v="
        parts = link.split(delim, 1)
        if len(parts) < 2: return None
        result = parts[1].split("&")[0]
        return result
    except Exception as e: logger.error(f"Error extracting video ID: {e}"); return None

class YTUrl(BaseModel):
    yt_url: str
    @validator("yt_url")
    def validate_yt_url(cls, v: str) -> str:
        if not v or not v.strip(): raise ValueError("Enter URL")
        v = v.strip()
        valid_domains = ["youtube.com", "m.youtube.com", "youtu.be"]
        if not any(d in v.lower() for d in valid_domains): raise ValueError("Enter valid YouTube URL")
        return v
# -----------------------------

# --- FastAPI App Definition ---
app = FastAPI(title="Kitchennaire Backend (URL Endpoint Only)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event(): logger.info("FastAPI server thread starting...")
@app.get("/")
async def root(): return {"status": "ok", "message": "Kitchennaire backend running"}
@app.post("/submit_url")
async def submit_url(payload: YTUrl):
    logger.info(f"API: Processing URL: {payload.yt_url}")
    try:
        videoId = getVideoId(payload.yt_url)
        if not videoId: raise ValueError("Could not extract video ID")
        logger.info(f"API: Extracted video ID: {videoId}")
        return {"status": "received", "yt_url": payload.yt_url, "video_id": videoId}
    except ValueError as e: return JSONResponse(status_code=422, content={"detail": str(e)})
@app.get("/get_latest_gesture")
async def get_latest_gesture():
    global latest_predicted_gesture, gesture_lock
    with gesture_lock: current_gesture = latest_predicted_gesture
    return {"status": "ok", "gesture": current_gesture}
# ------------------------------------

# --- Function to run FastAPI server ---
def run_api_server():
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e: logger.error(f"Failed to start API server: {e}")

# --- Main Execution Block (Local Webcam Loop - Mimics Reference Script) ---

if __name__ == "__main__":
    # 1. Load Model
    model = _load_weights()
    if model is None: exit()

    # 2. Start API Server Thread
    logger.info("Starting FastAPI server in background thread...")
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    time.sleep(3); logger.info("API server thread running.") if api_thread.is_alive() else logger.error("API thread failed.")

    # 3. Initialize Webcam
    accumWeight = ACCUM_WEIGHT
    camera = cv2.VideoCapture(0)
    if not camera.isOpened(): print("[FATAL] Could not open webcam."); exit()
    logger.info("Webcam opened successfully.")

    try: fps = int(camera.get(cv2.CAP_PROP_FPS)); fps = fps if 0 < fps < 200 else 30
    except Exception: fps = 30
    logger.info(f"Camera FPS set to: {fps}")

    top, right, bottom, left = ROI_COORDS['top'], ROI_COORDS['right'], ROI_COORDS['bottom'], ROI_COORDS['left']
    num_frames = 0
    k = 0 # Prediction counter
    # --- FIX: Store the last successful prediction ---
    last_valid_prediction = "Ready" # Start with Ready
    # -------------------------------------------------

    print("-" * 40 + "\nStarting Live Gesture Recognition (Webcam Mode)" + f"\nUsing Model: {MODEL_FILE_NAME}" + "\nAPI server running in background on port 8000." + "\nPress 'q' in the OpenCV window to exit." + "\n" + "-" * 40)

    # Webcam Loop
    while True:
        try:
            (grabbed, frame) = camera.read()
            if not grabbed: break

            frame = cv2.resize(frame, (700, 700)); frame = cv2.flip(frame, 1)
            clone = frame.copy()

            img_h, img_w = frame.shape[:2]
            t, r, b, l = max(0, top), max(0, right), min(img_h, bottom), min(img_w, left)
            if t >= b or r >= l: continue # Skip if ROI invalid
            roi = frame[t:b, r:l]
            if roi is None or roi.size == 0: continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0) # Use constant

            thresholded_display = None
            current_frame_prediction = last_valid_prediction # Default to last known good state

            if num_frames < CALIBRATION_FRAMES_REQUIRED:
                run_avg(gray, accumWeight)
                cv2.putText(clone, "CALIBRATING...", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                thresholded_display = gray
                current_frame_prediction = "Calibrating..."
                if num_frames == 1: print("[STATUS] please wait! calibrating...")
                elif num_frames == CALIBRATION_FRAMES_REQUIRED - 1:
                    print("[STATUS] calibration successfull...")
                    # --- FIX: Update shared state after calibration ---
                    last_valid_prediction = "Ready" # Reset after calibration
                    with gesture_lock: latest_predicted_gesture = "Ready"
                    # ------------------------------------------------

            else: # Segmentation and Prediction
                hand = segment(gray)
                thresholded_image = None # Initialize

                if hand is not None:
                    (thresholded, segmented) = hand
                    thresholded_display = thresholded
                    cv2.drawContours(clone, [segmented + (r, t)], -1, (0, 0, 255), 2)

                    prediction_interval = int(fps / 6) if fps > 0 else 5
                    if k % prediction_interval == 0:
                        if thresholded is not None and thresholded.size > 0:
                            cv2.imwrite('Temp.png', thresholded)
                            # --- FIX: Update only when predicting ---
                            predictedClassResult = getPredictedClass(model)
                            # Only update if it's not an error state
                            if "Error" not in predictedClassResult and "Waiting" not in predictedClassResult:
                                current_frame_prediction = predictedClassResult
                                last_valid_prediction = current_frame_prediction # Store the good prediction
                            else:
                                current_frame_prediction = last_valid_prediction # Keep last good state on error
                            # ---------------------------------------
                        else: # Thresholding failed
                            current_frame_prediction = "Seg. Error"
                            last_valid_prediction = current_frame_prediction # Store error state
                    else:
                        # --- FIX: Keep showing last valid prediction between intervals ---
                        current_frame_prediction = last_valid_prediction
                        # ------------------------------------------------------------
                else: # No hand detected
                    current_frame_prediction = "Blank"
                    last_valid_prediction = current_frame_prediction # Store blank state
                    roi_h, roi_w = b - t, l - r
                    thresholded_display = np.zeros((roi_h if roi_h>0 else 1, roi_w if roi_w>0 else 1), dtype=np.uint8)

                # Display prediction text (using current_frame_prediction)
                cv2.putText(clone, str(current_frame_prediction), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # --- UPDATE SHARED VARIABLE ---
            # Only update the shared variable if the prediction actually changed
            with gesture_lock:
                 if latest_predicted_gesture != current_frame_prediction:
                     latest_predicted_gesture = current_frame_prediction
            # ---------------------------

            # Display windows
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Video Feed - Hand Gesture Recognizer", clone)
            if thresholded_display is not None:
                 cv2.imshow("Thresholded Hand Segment", thresholded_display)

            num_frames += 1
            k += 1 # Use k like reference script

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'): break

        except KeyboardInterrupt: print("\nCtrl+C detected, stopping..."); break
        except Exception as loop_error:
            logger.error(f"Error in main loop: {loop_error}", exc_info=True)
            break

    # Cleanup
    print("[INFO] Releasing camera and closing windows...")
    camera.release()
    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)
    print("Application exited.")


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import logging
import base64
import io
import time # Import time for potential delays if needed

# --- 1. Machine Learning and Vision Imports ---
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
# ---------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kitchennaire-backend")

# Global variables and Constants for the ML system
ML_MODEL = None
bg = None # Global for background subtraction model
MODEL_FILE_NAME = "new_gesture_model.h5"

# Segmentation/Calibration Constants
CALIBRATION_FRAMES_REQUIRED = 60
ACCUM_WEIGHT_CALIBRATION = 1 / CALIBRATION_FRAMES_REQUIRED
# THRESHOLD_VALUE = 15 # REMOVED: Using Otsu's thresholding instead

# State variable to track calibration progress
CALIBRATION_FRAME_COUNT = 0


# --- ML Helper Functions ---

def run_avg(image, accum_weight):
    """Accumulates a weighted average of the background."""
    global bg
    # Robust Initialization Check
    if bg is None:
        if image is not None and image.size > 0: # Ensure image is valid before initializing
             bg = image.copy().astype("float")
             logger.info("--- BG initialized with first valid frame. ---")
        else:
             logger.error("--- Attempted to initialize BG with invalid image. ---")
             # Return False or raise an error if initialization fails? For now, just log.
        return

    # Only accumulate if image is valid
    if image is not None and image.size > 0:
        cv2.accumulateWeighted(image, bg, accum_weight)
    else:
        logger.warning("--- Skipping accumulateWeighted: invalid image received. ---")


def segment_hand(image_array): # Removed fixed threshold parameter
    """
    Segments the hand region from a frame array using Otsu's thresholding.
    Returns: (debug_image, ml_input_image, status)
    """
    global bg

    # 1. Preprocess the image array
    if image_array is None or image_array.size == 0:
        logger.error("--- Segmentation failed: Input image_array is invalid. ---")
        return (None, None, "ERROR")

    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
    except cv2.error as e:
        logger.error(f"--- OpenCV error during preprocessing: {e} ---")
        return (None, None, "ERROR")


    # Check if the background model is ready
    if bg is None:
        logger.info("--- Segmentation skipped: Background model (bg) is None. ---")
        # Return the gray image itself for debugging during calibration phase
        return (gray, None, "CALIBRATING")

    # Ensure background and gray image have same dimensions for absdiff
    if bg.shape != gray.shape:
        logger.warning(f"--- Shape mismatch: bg {bg.shape} vs gray {gray.shape}. Resetting bg. ---")
        bg = None # Force re-calibration
        return (gray, None, "CALIBRATING") # Indicate calibration needed

    # 2. Segmentation (Background Subtraction)
    try:
        diff = cv2.absdiff(bg.astype("uint8"), gray)

        # Apply Otsu's thresholding (automatically finds best threshold)
        ret, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"--- Otsu's Threshold Calculated: {ret} ---")

    except cv2.error as e:
        logger.error(f"--- OpenCV error during thresholding/absdiff: {e} ---")
        return (gray, None, "ERROR") # Return gray on error

    # --- Inversion Check (robust check for hand color) ---
    # Calculate average intensity inside potential hand region vs outside
    # This is complex, let's stick to the simpler check for now, but be aware it might fail
    if np.sum(thresholded == 255) > 0.8 * thresholded.size:
         thresholded = cv2.bitwise_not(thresholded)
         logger.info("--- Image Inversion Applied (Attempting Black Background). ---")
    # -----------------------------------------

    # Apply Dilation
    kernel = np.ones((4,4), np.uint8)
    try:
        thresholded = cv2.dilate(thresholded, kernel, iterations = 1)
    except cv2.error as e:
        logger.error(f"--- OpenCV error during dilation: {e} ---")
        # Continue without dilation if it fails
        pass


    # 3. Find and isolate the largest contour
    try:
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
         logger.error(f"--- OpenCV error during findContours: {e} ---")
         return (thresholded, None, "ERROR") # Return thresholded on error


    debug_image = thresholded.copy()

    if len(cnts) == 0:
        logger.info("--- Segmentation returned NO_HAND (No contours found). ---")
        return (debug_image, None, "NO_HAND")
    else:
        try:
            segmented_contour = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(segmented_contour)

            # Ensure bounding box has valid dimensions
            if w <= 0 or h <= 0:
                logger.warning("--- Invalid bounding box dimensions found. ---")
                return (debug_image, None, "NO_HAND")

            cropped_hand = thresholded[y:y+h, x:x+w]

            # Ensure cropped hand is not empty before resizing
            if cropped_hand is None or cropped_hand.size == 0:
                 logger.warning("--- Cropped hand image is empty. ---")
                 return (debug_image, None, "NO_HAND")

            hand_image_for_model = cv2.resize(cropped_hand, (100, 120))
            logger.info(f"Hand segmented and resized successfully. Hand area: {w*h} pixels.")

        except cv2.error as e:
             logger.error(f"--- OpenCV error during cropping/resizing: {e} ---")
             return (debug_image, None, "ERROR")
        except Exception as e:
             logger.error(f"--- Unexpected error during contour processing: {e} ---")
             return (debug_image, None, "ERROR")


        return (debug_image, hand_image_for_model, "READY")

def get_predicted_class(model, hand_image):
    """Runs model inference on the segmented hand image."""
    if hand_image is None:
        logger.error("--- Prediction skipped: hand_image is None ---")
        return "Error"

    try:
        # Ensure image is single channel before reshape
        if len(hand_image.shape) > 2 and hand_image.shape[2] != 1:
             logger.warning(f"--- Hand image is not grayscale ({hand_image.shape}), attempting conversion. ---")
             # This shouldn't happen if segmentation is correct, but as a fallback
             hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)


        # Final check on dimensions before reshape
        if hand_image.shape != (120, 100):
             logger.warning(f"--- Hand image incorrect shape before reshape: {hand_image.shape}. Attempting resize again. ---")
             hand_image = cv2.resize(hand_image, (100, 120)) # Force resize

        model_input = hand_image.reshape(1, 100, 120, 1)

    except ValueError as e:
         logger.error(f"--- Reshape failed: {e}. Hand image shape: {hand_image.shape} ---")
         return "Error"
    except Exception as e:
         logger.error(f"--- Unexpected error during prediction preparation: {e} ---")
         return "Error"


    try:
        prediction = model.predict_on_batch(model_input)
        predicted_class = np.argmax(prediction)
    except Exception as e:
        logger.error(f"--- Error during model.predict_on_batch: {e} ---")
        return "Error"


    # Map the index to the gesture label
    if predicted_class == 0:    label = "Blank"
    elif predicted_class == 1:  label = "OK"
    elif predicted_class == 2:  label = "Thumbs Up"
    elif predicted_class == 3:  label = "Thumbs Down"
    elif predicted_class == 4:  label = "Punch"
    elif predicted_class == 5:  label = "High Five"
    else:                       label = "Unknown"

    logger.info(f"--- PREDICTION RESULT: {label} (Index: {predicted_class}) ---")
    return label

def _load_weights():
    """Loads the ML model once at startup."""
    custom_objects = {'BatchNormalization': BatchNormalization}
    try:
        model = load_model(MODEL_FILE_NAME, custom_objects=custom_objects)
        logger.info(f"Model {MODEL_FILE_NAME} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"[FATAL] Model loading failed: {e}")
        return None

# --- YouTube URL Handlers --- (Keep your existing functions)
def getVideoId(link):
    print(f"Attempting to extract video ID from: {link}")
    try:
        link = link.replace("m.youtube.com", "youtube.com")
        if "youtu.be" in link: delim = "youtu.be/"
        else: delim = "watch?v="
        parts = link.split(delim, 1)
        if len(parts) < 2:
            print(f"ERROR: Delimiter not found: {link}")
            return None
        result = parts[1].split("&")[0]
        print(f"Extracted video ID: {result}")
        return result
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

class YTUrl(BaseModel):
    yt_url: str
    @validator("yt_url")
    def validate_yt_url(cls, v: str) -> str:
        print(f"Button Pressed: {v}")
        if not v or not v.strip(): raise ValueError("Enter URL")
        v = v.strip()
        valid_domains = ["youtube.com", "m.youtube.com", "youtu.be"]
        if not any(d in v.lower() for d in valid_domains):
            print(f"ERROR: Not valid YT link: {v}")
            raise ValueError("Enter valid YouTube URL")
        return v
# -----------------------------

# --- FastAPI App Initialization ---
app = FastAPI(title="Kitchennaire Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global ML_MODEL
    ML_MODEL = _load_weights()
    print("Backend running and model loaded.")

# --- FastAPI Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Kitchennaire backend running"}

@app.post("/submit_url")
async def submit_url(payload: YTUrl):
    try:
        yt_url = payload.yt_url
        logger.info(f"Processing URL: {yt_url}")
        videoId = getVideoId(yt_url)
        if not videoId: raise ValueError("Could not extract video ID")
        logger.info(f"Extracted video ID: {videoId}")
        return {"status": "received", "yt_url": yt_url, "video_id": videoId}
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        return JSONResponse(status_code=422, content={"detail": str(e)})

# --- Function to Safely Show/Update Debug Window ---
# Wrap imshow calls to prevent crashes in server environment
_debug_windows_open = set()
def safe_imshow(window_name, image):
    global _debug_windows_open
    try:
        if image is None or image.size == 0:
            logger.warning(f"Attempted to show empty image in window '{window_name}'.")
            return
        cv2.imshow(window_name, image)
        _debug_windows_open.add(window_name) # Track open windows
        cv2.waitKey(1) # Crucial for window refresh
    except cv2.error as e:
        logger.error(f"OpenCV error showing window '{window_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error showing window '{window_name}': {e}")

# --- Function to Safely Close Debug Windows ---
def safe_destroyAllWindows():
    global _debug_windows_open
    logger.info("Attempting to close debug windows.")
    try:
        for window_name in list(_debug_windows_open): # Iterate over a copy
             cv2.destroyWindow(window_name)
             _debug_windows_open.remove(window_name)
        logger.info("Debug windows closed.")
    except cv2.error as e:
        logger.error(f"OpenCV error closing windows: {e}")
        _debug_windows_open.clear() # Clear tracking even on error
    except Exception as e:
        logger.error(f"Unexpected error closing windows: {e}")
        _debug_windows_open.clear()

@app.post("/predict_gesture")
async def predict_gesture(
    image: UploadFile = File(...),
    is_calibrating: str = Form(...)
):
    global ML_MODEL, bg, CALIBRATION_FRAME_COUNT

    logger.info(f"Request received. Calibrating: {is_calibrating}. Frame Count: {CALIBRATION_FRAME_COUNT}")

    if ML_MODEL is None:
        logger.error("Predicting failed: ML_MODEL is None.")
        return JSONResponse(status_code=503, content={"detail": "ML model not loaded."})

    debug_window_name_calib = "Server Segmentation Debug (Calibrating)"
    debug_window_name_pred = "Server Segmentation Debug (Predicting)"

    try:
        # 1. Read and decode image
        image_bytes = await image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_array is None:
            logger.error("Image decoding failed.")
            raise ValueError("Could not decode image.")

        logger.info(f"Image decoded. Shape: {frame_array.shape}. Size: {len(image_bytes)} bytes.")

        # --- ROI DEFINITION (ENLARGED) ---
        top, right, bottom, left = 100, 100, 650, 650
        # Robust ROI Boundary Check
        img_h, img_w = frame_array.shape[:2]
        top = max(0, top)
        right = max(0, right)
        bottom = min(img_h, bottom)
        left = min(img_w, left)
        if top >= bottom or right >= left:
             logger.error(f"ROI coordinates [{top}:{bottom}, {right}:{left}] are invalid after clamping for image shape {frame_array.shape}. Using full image.")
             roi = frame_array # Use full image as fallback
        else:
             roi = frame_array[top:bottom, right:left]
        # --------------------------------

        # 2. Handle Calibration State
        is_calibrating_bool = is_calibrating.lower() == 'true'

        if is_calibrating_bool and CALIBRATION_FRAME_COUNT == 0:
            bg = None
            logger.info("--- CALIBRATION SEQUENCE STARTING (bg reset). ---")
            safe_destroyAllWindows() # Close prediction window if starting calibration

        # Calibration Logic
        if bg is None or CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_REQUIRED:
            if roi is None or roi.size == 0:
                 logger.warning("ROI is empty during calibration, skipping frame.")
                 return {"status": "calibrating", "gesture": f"Calibrating (Bad ROI): {CALIBRATION_FRAME_COUNT}/{CALIBRATION_FRAMES_REQUIRED}"}

            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
            except cv2.error as e:
                 logger.error(f"OpenCV error during calibration preprocessing: {e}")
                 # Still return calibrating status, but don't increment
                 return {"status": "calibrating", "gesture": f"Calibrating (Preproc Error): {CALIBRATION_FRAME_COUNT}/{CALIBRATION_FRAMES_REQUIRED}"}


            run_avg(gray, ACCUM_WEIGHT_CALIBRATION)
            CALIBRATION_FRAME_COUNT += 1

            # Safely show debug window
            if bg is not None:
                debug_frame = cv2.convertScaleAbs(bg)
                safe_imshow(debug_window_name_calib, debug_frame)

            if CALIBRATION_FRAME_COUNT >= CALIBRATION_FRAMES_REQUIRED:
                 safe_destroyAllWindows()
                 logger.info("--- CALIBRATION COMPLETE! ---")
                 return {"status": "ready", "gesture": "Calibration Complete"}
            else:
                 return {"status": "calibrating", "gesture": f"Calibrating: {CALIBRATION_FRAME_COUNT}/{CALIBRATION_FRAMES_REQUIRED}"}

        # 3. Run Segmentation and Prediction (Post-Calibration)
        if roi is None or roi.size == 0:
            logger.warning("ROI is empty during prediction, skipping frame.")
            return {"status": "ready", "gesture": "Blank"}

        debug_image, ml_input_image, status = segment_hand(roi)

        # Safely show debug window
        safe_imshow(debug_window_name_pred, debug_image)

        if status == "NO_HAND" or status == "ERROR" or ml_input_image is None:
             # If segmentation failed or returned error, predict Blank
            return {"status": "ready", "gesture": "Blank"}

        predicted_gesture = get_predicted_class(ML_MODEL, ml_input_image)

        return {"status": "predicting", "gesture": predicted_gesture}

    except ValueError as ve:
        logger.error(f"ValueError in /predict_gesture: {ve}")
        safe_destroyAllWindows()
        return JSONResponse(status_code=400, content={"detail": str(ve)})
    except Exception as e:
        logger.error(f"FATAL Error in /predict_gesture: {e}", exc_info=True)
        safe_destroyAllWindows()
        return JSONResponse(status_code=500, content={"detail": "Server error."})

# Add a shutdown event to ensure windows are closed when server stops
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down FastAPI server.")
    safe_destroyAllWindows()


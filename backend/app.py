from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import logging
import base64
import io

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

# Segmentation/Calibration Constants (Optimized based on discussion)
CALIBRATION_FRAMES_REQUIRED = 30 
ACCUM_WEIGHT_CALIBRATION = 1 / CALIBRATION_FRAMES_REQUIRED # Effective weight for 30 frames
THRESHOLD_VALUE = 15 # Optimized threshold for segmentation sensitivity

# State variable to track calibration progress across stateless API calls
CALIBRATION_FRAME_COUNT = 0 


# --- ML Helper Functions (Integrated from your original script) ---

def run_avg(image, accumWeight):
    """Accumulates a weighted average of the background."""
    global bg
    if bg is None:
        # Initialize the background model with the first frame
        bg = image.copy().astype("float")
        return
    # Accumulate the weighted average
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment_hand(image_array, threshold=THRESHOLD_VALUE):
    """Segments the hand region from a frame array."""
    global bg
    
    # 1. Preprocess the image array for segmentation
    # We apply blur here to match the logic in the calibration step
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0) # Increased blur
    
    # Check if the background model is ready
    if bg is None:
        return (None, "CALIBRATING") 
        
    # 2. Segmentation (Background Subtraction)
    diff = cv2.absdiff(bg.astype("uint8"), gray)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Apply Dilation to fill small holes and improve contour
    kernel = np.ones((4,4), np.uint8) 
    thresholded = cv2.dilate(thresholded, kernel, iterations = 1)
    
    # 3. Find and isolate the largest contour
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return (None, "NO_HAND")
    else:
        # Extract the hand region from the thresholded image based on the largest contour
        segmented_contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(segmented_contour)
        
        # Crop and resize the final hand image to (100, 120) for the model
        hand_image = thresholded[y:y+h, x:x+w]
        
        # We need to handle the case where cropping might result in zero dimensions, 
        # though resizing should ensure it matches the model input
        try:
             hand_image = cv2.resize(hand_image, (100, 120))
        except cv2.error:
             return (None, "NO_HAND") # Return no hand if resize fails

        return (hand_image, "READY")

def get_predicted_class(model, hand_image):
    """Runs model inference on the segmented hand image."""
    # Reshape for model: (batch, H, W, channels=1)
    model_input = hand_image.reshape(1, 100, 120, 1)

    prediction = model.predict_on_batch(model_input)
    predicted_class = np.argmax(prediction)
    
    # Map the index to the gesture label
    if predicted_class == 0:
        return "Blank"
    elif predicted_class == 1:
        return "OK"
    elif predicted_class == 2:
        return "Thumbs Up"
    elif predicted_class == 3:
        return "Thumbs Down"
    elif predicted_class == 4:
        return "Punch"
    elif predicted_class == 5:
        return "High Five"
    return "Unknown"

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

# --- YouTube URL Handlers (from your existing file) ---

def getVideoId(link):
    """Gets the video Id from a valid yt link"""
    try:
        link = link.replace("m.youtube.com", "youtube.com")
        
        if "youtu.be" in link:
            delim = "youtu.be/"
        else:
            delim = "watch?v="
        
        parts = link.split(delim, 1)
        if len(parts) < 2:
            return None
            
        result = parts[1]
        if "&" in result:
            result = result.split("&")[0]
            
        return result
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        return None

class YTUrl(BaseModel):
    yt_url: str

    @validator("yt_url")
    def validate_yt_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Please enter a URL")
        v = v.strip()
        
        valid_domains = ["youtube.com", "m.youtube.com", "youtu.be"]
        if not any(domain in v.lower() for domain in valid_domains):
            raise ValueError("Please enter a valid YouTube URL (must contain youtube.com or youtu.be)")
        return v
    
# --- FastAPI App Initialization ---

app = FastAPI(title="Kitchennaire Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Loads the ML model into memory when the server starts."""
    global ML_MODEL
    ML_MODEL = _load_weights()
    print("Backend running and model loaded.")

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "Kitchennaire backend running"}


@app.post("/submit_url")
async def submit_url(payload: YTUrl):
    """Receives a YouTube URL, validates it, and extracts the video ID."""
    try:
        yt_url = payload.yt_url
        logger.info(f"Processing URL: {yt_url}")
        videoId = getVideoId(yt_url)
        
        if not videoId:
            raise ValueError("Could not extract video ID from URL")
            
        logger.info(f"Extracted video ID: {videoId}")
        return {"status": "received", "yt_url": yt_url, "video_id": videoId}
    except ValueError as e:
        logger.error("Invalid payload received: %s", str(e))
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(e),
                "help": "Ensure you're sending a JSON object with a 'yt_url' field containing a YouTube URL"
            }
        )

@app.post("/predict_gesture")
async def predict_gesture(
    image: UploadFile = File(...), 
    is_calibrating: str = Form(...) # Expect "true" or "false" from Expo
):
    """
    Receives an image (frame) from the mobile application and returns a gesture prediction.
    """
    global ML_MODEL
    global bg
    global CALIBRATION_FRAME_COUNT 
    
    if ML_MODEL is None:
        return JSONResponse(status_code=503, content={"detail": "ML model not loaded."})

    try:
        # 1. Read and decode the image (assuming JPEG or PNG from Expo)
        image_bytes = await image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_array is None:
            raise ValueError("Could not decode image.")

        # Using the same ROI size as your original script for all processing:
        top, right, bottom, left = 10, 350, 225, 590
        roi = frame_array[top:bottom, right:left]
        
        # 2. Handle Calibration State (Stateless API Logic)
        is_calibrating_bool = is_calibrating.lower() == 'true'
        
        # Reset background model if the client requests calibration or the model is empty
        if is_calibrating_bool and CALIBRATION_FRAME_COUNT == 0:
            bg = None
            logger.info("Starting new calibration sequence.")

        # If calibration is active (bg is empty or we are tracking frames)
        if bg is None or CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_REQUIRED:
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)
            
            # Accumulate the background model
            run_avg(gray, ACCUM_WEIGHT_CALIBRATION)
            CALIBRATION_FRAME_COUNT += 1
            
            # Check if calibration is finished
            if CALIBRATION_FRAME_COUNT >= CALIBRATION_FRAMES_REQUIRED:
                logger.info("Calibration sequence complete.")
                # We do NOT reset CALIBRATION_FRAME_COUNT here; it stays at 30 until the client forces a reset.
                return {"status": "ready", "gesture": "Calibration Complete"}
            else:
                return {"status": "calibrating", "gesture": f"Calibrating: {CALIBRATION_FRAME_COUNT}/{CALIBRATION_FRAMES_REQUIRED}"}

        # 3. Run Segmentation and Prediction (Only when calibration is complete)
        
        hand_image, status = segment_hand(roi, THRESHOLD_VALUE)

        if status == "NO_HAND":
            return {"status": "ready", "gesture": "Blank"}

        # Run model inference
        predicted_gesture = get_predicted_class(ML_MODEL, hand_image)
        
        return {"status": "predicting", "gesture": predicted_gesture}

    except Exception as e:
        logger.error(f"Error in /predict_gesture: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Server-side processing error."}
        )
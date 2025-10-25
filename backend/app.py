from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kitchennaire-backend")


def getVideoId(link):
    """Gets the video Id from a valid yt link"""
    try:
        # Handle mobile URLs
        link = link.replace("m.youtube.com", "youtube.com")
        
        if "youtu.be" in link:
            delim = "youtu.be/" # short links only have stuff after /
        else:
            delim = "watch?v=" # long links have stuff after =
        
        parts = link.split(delim, 1)
        if len(parts) < 2:
            print(f"ERROR: Could not find video ID in URL: {link}")
            return None
            
        result = parts[1]
        if "&" in result: # handle timestamps and other parameters
            result = result.split("&")[0]
            
        print(f"Extracted video ID: {result}")
        return result
    except Exception as e:
        print(f"Error extracting video ID: {str(e)}")
        return None

class YTUrl(BaseModel):
    yt_url: str

    @validator("yt_url")
    def validate_yt_url(cls, v: str) -> str:
        print(f"Button Pressed: {v}")

        if not v or not v.strip():
            raise ValueError("Please enter a URL")
        v = v.strip()
        
        # Basic YouTube URL validation
        valid_domains = ["youtube.com", "m.youtube.com", "youtu.be"]
        if not any(domain in v.lower() for domain in valid_domains):
            print(f"ERROR: Not a valid YT link")
            raise ValueError("Please enter a valid YouTube URL (must contain youtube.com or youtu.be)")
        return v
    

app = FastAPI(title="Kitchennaire Backend")

# Development CORS: allow all origins. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Kitchennaire backend running"}


@app.on_event("startup")
async def startup_event():
    # Simple startup log so you can see when the backend launches
    print("Backend running")


@app.post("/submit_url")
async def submit_url(payload: YTUrl):
    try:
        yt_url = payload.yt_url
        print(f"Processing URL: {yt_url}")
        videoId = getVideoId(yt_url)
        
        if not videoId:
            raise ValueError("Could not extract video ID from URL")
            
        print(f"Extracted video ID: {videoId}")
        return {"status": "received", "yt_url": yt_url, "video_id": videoId}
    except ValueError as e:
        logger.error("Invalid payload received: %s", str(e))
        print(f"Error processing payload: {str(e)}")
        # Return a more helpful error message
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(e),
                "help": "Ensure you're sending a JSON object with a 'yt_url' field containing a YouTube URL"
            }
        )

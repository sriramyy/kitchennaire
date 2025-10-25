from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kitchennaire-backend")


def getVideoId(link):
    """Gets the video Id from a valid yt link"""

    result = ""

    if "youtu.be" in link:
        delim = "youtu.be/" # short links only have stuff after /
    else:
        delim = "watch?v=" # long links have stuff after =
    
    parts = link.split(delim, 1)

    if "&" in parts[1]: # do it again incase there is time stamp in link
        delim2 = "&" 
        parts2 = parts[1].split(delim2, 1)
        result = parts2[0]
    else:
        result = parts[1]
    


    print(result)

class YTUrl(BaseModel):
    yt_url: str

    @validator("yt_url")
    def validate_yt_url(cls, v: str) -> str:

        print(f"Button Pressed: {v}")

        if not v or not v.strip():
            raise ValueError("Please enter a URL")
        v = v.strip()
        
        # Basic YouTube URL validation
        valid_domains = ["youtube.com", "youtu.be"]
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
        print(f"Correct yt_url: {yt_url}")
        videoId = getVideoId(yt_url)
        return {"status": "received", "yt_url": yt_url}
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

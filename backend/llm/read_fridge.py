# read_fridge.py (OpenAI SDK 2.x)
import os
import json
import base64
import openai as openai_pkg  # just to print version
from dotenv import load_dotenv
from openai import OpenAI

# --- Config: hardcode your image file ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "photo.jpg")  # ensure the filename matches exactly

# --- Load API key ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY not found. Put it in a .env file next to this script.")

client = OpenAI(api_key=API_KEY)

# --- Helper: convert local image to data URL ---
def to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No such file: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        raise ValueError("Use .jpg, .jpeg, or .png images only.")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

SYSTEM_PROMPT = (
    "You are a careful, literal visual inventory assistant. "
    "From a single photo (fridge/pantry/cabinet/counter), list ONLY visible food/beverage/condiment items. "
    "Do not hallucinate. If uncertain about an item, include it with lower confidence."
)

USER_INSTRUCTIONS = (
    "List every visible consumable item. Prefer generic names (e.g., 'milk', 'egg', 'butter'). "
    "If a brand label is clearly readable, include it. "
    "Estimate quantity ONLY if it's obvious (e.g., visible egg count). "
    "Return STRICT JSON only with fields: name, quantity_estimate|null, unit|null, brand_or_label|null, confidence (0..1). "
    "Also include a top-level 'notes' string|null."
)

def read_image_items(image_path: str, model: str = "gpt-4o-mini") -> dict:
    data_url = to_data_url(image_path)

    # IMPORTANT: For openai 2.x, image must be an object under image_url: {"url": ...}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_INSTRUCTIONS},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Model returned non-JSON", "raw": raw}

if __name__ == "__main__":
    try:
        print("OpenAI SDK version:", openai_pkg.__version__)
        print("Analyzing image:", IMAGE_PATH)
        result = read_image_items(IMAGE_PATH)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # ✅ ADD THIS SECTION (still inside the try!)
        PANTRY_JSON = os.path.join(SCRIPT_DIR, "pantry.json")
        with open(PANTRY_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved pantry JSON to: {PANTRY_JSON}")

    except Exception as e:
        print(f"ERROR: {e}")


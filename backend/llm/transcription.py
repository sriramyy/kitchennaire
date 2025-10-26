# transcription.py — fast + Whisper-free
# -----------------------------------------------------------------------------
# URL prompt → get transcript quickly:
#   1) youtube-transcript-api
#   2) yt_dlp subtitles (manual then auto) parsed from .vtt
#   3) chapters/description timestamps (fallback for timing)
# Output JSON: title, total_time_minutes, serves, video_duration_(seconds|hms),
# ingredients[{name,quantity,unit,notes}], steps_detailed with H:MM:SS ranges.
# -----------------------------------------------------------------------------

import os
import re
import json
import difflib
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yt_dlp

from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# -------------------- CONFIG --------------------
CHAT_MODEL = "gpt-4o-mini"  # LLM for recipe structuring

# -------------------- SETUP ---------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY not found in .env")
client = OpenAI(api_key=API_KEY)

# -------------------- UTILS ---------------------
def seconds_to_hms(s: Optional[float]) -> Optional[str]:
    if not isinstance(s, (int, float)):
        return None
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"

def parse_timestamp_to_seconds(ts: str) -> Optional[int]:
    m = re.match(r'^(?:(\d+):)?(\d{1,2}):(\d{2})$', ts.strip())
    if not m:
        return None
    h = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2)); ss = int(m.group(3))
    return h*3600 + mm*60 + ss

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s:/.-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------- VTT PARSER (minimal) --------------------
def parse_vtt(vtt_text: str) -> List[Dict]:
    """
    Parse WebVTT into segments: [{"start": seconds, "end": seconds, "text": str}, ...]
    Very lightweight; ignores cues like NOTE, STYLE.
    """
    segs = []
    start = end = None
    buf: List[str] = []
    for line in vtt_text.splitlines():
        line = line.rstrip("\n")
        if "-->" in line:
            # flush previous
            if start is not None and buf:
                segs.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(t for t in buf if t).strip()
                })
                buf = []
            # time line: 00:01.000 --> 00:03.000
            tm = re.match(r'.*?(\d+:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*(\d+:\d{2}(?::\d{2})?\.\d{3})', line)
            def to_sec(hms_ms: str) -> int:
                # accepts MM:SS.mmm or H:MM:SS.mmm
                parts = hms_ms.split(".")[0].split(":")
                parts = [int(p) for p in parts]
                if len(parts) == 2:
                    m, s = parts
                    return m*60 + s
                if len(parts) == 3:
                    h, m, s = parts
                    return h*3600 + m*60 + s
                return 0
            if tm:
                start = to_sec(tm.group(1))
                end = to_sec(tm.group(2))
            else:
                start = end = None
        elif line.strip() == "" or line.strip().startswith(("WEBVTT", "NOTE", "STYLE")):
            continue
        else:
            buf.append(line.strip())

    if start is not None and buf:
        segs.append({
            "start": start,
            "end": end,
            "text": " ".join(t for t in buf if t).strip()
        })
    return segs

# -------------------- YOUTUBE HELPERS -------------------
def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|/v/|/embed/|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def get_transcript_text_and_segments(video_id: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Primary: youtube-transcript-api (fast). Prefers human 'en', then auto 'en', else first available.
    Returns (full_text, segments[{text,start,duration}]).
    """
    try:
        tl = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred = None
        for t in tl:
            if t.language_code.startswith("en") and not t.is_generated:
                preferred = t; break
        if preferred is None:
            for t in tl:
                if t.language_code.startswith("en"):
                    preferred = t; break
        if preferred is None:
            preferred = next(iter(tl), None)
        if preferred is None:
            return None, None
        segs = preferred.fetch()
        text = " ".join(ch["text"] for ch in segs if ch["text"].strip())
        return (text, segs)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None, None
    except Exception:
        return None, None

def fetch_metadata(url: str, cookies: Optional[str] = None) -> Dict:
    """yt_dlp metadata only (no download), optional cookies file."""
    opts = {"quiet": True, "noprogress": True, "skip_download": True}
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)

def get_subs_with_ytdlp(url: str, cookies: Optional[str] = None) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Fast subtitle fetch via yt_dlp:
      - try manual English subtitles first
      - then auto English subtitles
    Returns (full_text, segments[{text,start,duration}]) or (None, None).
    """
    tmp = Path(tempfile.mkdtemp(prefix="subs_"))
    # Make yt_dlp write subs to our temp folder without downloading the video
    opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": True,
        "writesubtitles": True,       # manual
        "writeautomaticsub": True,    # auto
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
        "paths": {"home": str(tmp)},
        "outtmpl": {"subtitle": "%(id)s.%(ext)s"},
    }
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies

    # Step 1: trigger subtitle write
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        vid = info.get("id")

        # If yt_dlp found URLs for subs, we can force download the sub files now
        has_subs = bool(info.get("subtitles") or info.get("automatic_captions"))
        if has_subs:
            ydl.download([url])

    # Step 2: locate VTT files in tmp
    vtts = list(tmp.glob("*.vtt"))
    if not vtts:
        return None, None

    # Prefer manual subs over auto by filename heuristic (auto often contains ".en" with "auto" or similar)
    def manual_first(p: Path) -> int:
        name = p.name.lower()
        if "auto" in name:
            return 1
        return 0
    vtts.sort(key=manual_first)

    # Parse the first available .vtt into segments and text
    vtt_text = vtts[0].read_text(encoding="utf-8", errors="ignore")
    segs = parse_vtt(vtt_text)
    if not segs:
        return None, None
    full_text = " ".join(s["text"] for s in segs if s.get("text"))
    # adapt to common segment shape we use elsewhere
    segments = [{"text": s["text"], "start": s["start"], "duration": max(0, (s.get("end") or s["start"]) - s["start"])} for s in segs]
    return full_text, segments

# -------------------- LLM: RECIPE EXTRACTION --------------------
def extract_recipe(transcript: str) -> Dict:
    SYSTEM = (
        "You are a meticulous recipe extractor. "
        "Given a cooking video transcript, output ONLY valid JSON. "
        "Infer conservatively; use null when unknown."
    )
    PROMPT = (
        "From this transcript, extract a clean recipe. "
        "Return ONLY JSON:\n"
        "{\n"
        '  \"title\": \"string|null\",\n'
        '  \"total_time_minutes\": number|null,\n'
        '  \"serves\": number|null,\n'
        '  \"ingredients\": [\n'
        '    {\"name\":\"string\",\"quantity\":\"string|null\",\"unit\":\"string|null\",\"notes\":\"string|null\"}\n'
        "  ],\n"
        '  \"steps_detailed\": [\n'
        '    {\"step\": number, \"instruction\": \"string\", \"duration_minutes\": number|null, \"temperature_c\": number|null, \"temperature_f\": number|null}\n'
        "  ]\n"
        "}\n\n"
        "Rules: include ALL ingredients; keep ranges/fractions as strings; steps ordered and actionable; "
        "total_time_minutes includes prep + cook if possible; use null when unstated.\n\n"
        f"Transcript:\n{transcript}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("ingredients", [])
    data.setdefault("steps_detailed", [])
    return data

# -------------------- TIMING FROM CHAPTERS / DESCRIPTION -------------
def build_ranges_from_chapters(chapters: List[Dict], video_duration: Optional[int]) -> List[Tuple[int,int,str]]:
    out = []
    pairs = sorted([(int(c.get("start_time", 0)), c.get("title", "") or "") for c in chapters], key=lambda x: x[0])
    for i, (s, title) in enumerate(pairs):
        e = (pairs[i+1][0] - 1) if i < len(pairs) - 1 else ((video_duration - 1) if isinstance(video_duration, int) else None)
        if isinstance(e, int) and e < s:
            e = None
        out.append((s, e, title))
    return out

def build_ranges_from_description(description: str, video_duration: Optional[int]) -> List[Tuple[int,int,str]]:
    stamps = []
    for line in description.splitlines():
        m = re.search(r'((?:\d+:)?\d{1,2}:\d{2})\s+[-–—:]?\s*(.+)$', line.strip())
        if m:
            sec = parse_timestamp_to_seconds(m.group(1))
            if sec is not None:
                stamps.append((sec, m.group(2).strip()))
    stamps = sorted(stamps, key=lambda x: x[0])
    out = []
    for i, (s, title) in enumerate(stamps):
        e = (stamps[i+1][0] - 1) if i < len(stamps) - 1 else ((video_duration - 1) if isinstance(video_duration, int) else None)
        if isinstance(e, int) and e < s:
            e = None
        out.append((s, e, title))
    return out

def assign_steps_even_split(steps: List[Dict], ranges: List[Tuple[int,int,str]]) -> List[Dict]:
    """
    Map each polished step to the best range by title similarity; if multiple steps map
    to the same range, evenly split that range across those steps (ordered).
    """
    if not steps or not ranges:
        return steps

    def score(instr: str, title: str) -> float:
        a = _normalize_text(instr); b = _normalize_text(title)
        base = difflib.SequenceMatcher(a=a, b=b).ratio()
        for kw in ["preheat","mix","whisk","fold","cream","bake","simmer","boil","knead","rest","frost","assemble","pour","divide"]:
            if kw in a and kw in b:
                base += 0.1
        return base

    bins: Dict[int, List[int]] = {i: [] for i in range(len(ranges))}
    for idx, st in enumerate(steps):
        instr = st.get("instruction", "") or ""
        best_i, best_s = 0, -1.0
        for ri, (_, _, title) in enumerate(ranges):
            sc = score(instr, title)
            if sc > best_s:
                best_s = sc; best_i = ri
        bins[best_i].append(idx)

    for ri, step_idxs in bins.items():
        s, e, _ = ranges[ri]
        if not isinstance(s, int):
            for si in step_idxs:
                steps[si]["start_seconds"] = None
                steps[si]["end_seconds"] = None
                steps[si]["start_hms"] = None
                steps[si]["end_hms"] = None
                steps[si]["timestamp_range"] = None
            continue

        if e is None or e < s:
            for si in step_idxs:
                steps[si]["start_seconds"] = s
                steps[si]["end_seconds"] = None
                steps[si]["start_hms"] = seconds_to_hms(s)
                steps[si]["end_hms"] = None
                steps[si]["timestamp_range"] = f"{steps[si]['start_hms']}–?"
            continue

        count = max(1, len(step_idxs))
        total = max(1, e - s + 1)
        slice_len = total // count

        for j, si in enumerate(step_idxs):
            start = s + j * slice_len
            end = e if j == count - 1 else (s + (j + 1) * slice_len - 1)
            if end < start:
                end = start
            steps[si]["start_seconds"] = int(start)
            steps[si]["end_seconds"] = int(end)
            steps[si]["start_hms"] = seconds_to_hms(start)
            steps[si]["end_hms"] = seconds_to_hms(end)
            steps[si]["timestamp_range"] = f"{steps[si]['start_hms']}–{steps[si]['end_hms']}"

    return steps

def align_steps_to_segments_simple(steps: List[Dict], segments: List[Dict]) -> List[Dict]:
    """
    If no chapters/description ranges, align to transcript segments (fast).
    """
    if not steps or not segments:
        return steps
    segs = [{"start": int(s.get("start", 0)), "text": _normalize_text(s.get("text", ""))} for s in segments]
    for st in steps:
        instr = _normalize_text(st.get("instruction", ""))
        best, best_s = None, 0.0
        for seg in segs:
            sc = difflib.SequenceMatcher(a=instr, b=seg["text"]).ratio()
            if sc > best_s:
                best, best_s = seg, sc
        if best and best_s >= 0.25:
            st["start_seconds"] = best["start"]
            st["start_hms"] = seconds_to_hms(best["start"])
        else:
            st.setdefault("start_seconds", None)
            st.setdefault("start_hms", None)
    return steps

def fill_end_times_from_next_or_duration(steps: List[Dict]) -> List[Dict]:
    starts = [s.get("start_seconds") for s in steps]
    for i, st in enumerate(steps):
        start = st.get("start_seconds")
        end = None
        if isinstance(start, int):
            later = [s for idx, s in enumerate(starts) if idx > i and isinstance(s, int) and s > start]
            if later:
                end = min(later) - 1
        if isinstance(end, int) and isinstance(start, int) and end < start:
            end = None
        st["end_seconds"] = end
        st["start_hms"] = seconds_to_hms(start) if isinstance(start, int) else None
        st["end_hms"] = seconds_to_hms(end) if isinstance(end, int) else None
        st["timestamp_range"] = (
            f"{st['start_hms']}–{st['end_hms']}" if st["start_hms"] and st["end_hms"]
            else (f"{st['start_hms']}–?" if st["start_hms"] else None)
        )
    return steps

# -------------------- MAIN PIPELINE ---------------------------
def youtube_to_recipe(url: str, cookies: Optional[str] = None) -> Dict:
    vid = extract_video_id(url)
    if not vid:
        raise ValueError("Invalid YouTube URL")

    # 1) Try youtube-transcript-api (fast)
    transcript_text, segments = get_transcript_text_and_segments(vid)

    # 2) If that fails, try yt_dlp subtitles (manual → auto), still fast
    if not transcript_text or len(transcript_text) < 10:
        transcript_text, segments = get_subs_with_ytdlp(url, cookies=cookies)

    if not transcript_text or not segments:
        raise SystemExit(
            "No transcript/subtitles accessible.\n"
            "Tips:\n"
            "  • Pass cookies from your browser (age/member/region): --cookies /path/to/cookies.txt\n"
            "  • Or pick another video with captions enabled."
        )

    # 3) Extract recipe structure
    recipe = extract_recipe(transcript_text)
    steps = recipe.get("steps_detailed") or []

    # 4) Metadata: duration/chapters/description (no download)
    meta = fetch_metadata(url, cookies=cookies)
    video_duration = meta.get("duration") if isinstance(meta.get("duration"), int) else None
    chapters = meta.get("chapters") or []
    description = meta.get("description") or ""

    # 5) Prefer chapter ranges → description → segment similarity
    if chapters:
        ranges = build_ranges_from_chapters(chapters, video_duration)
        steps = assign_steps_even_split(steps, ranges)
    else:
        desc_ranges = build_ranges_from_description(description, video_duration)
        if desc_ranges:
            steps = assign_steps_even_split(steps, desc_ranges)
        else:
            steps = align_steps_to_segments_simple(steps, segments)
            steps = fill_end_times_from_next_or_duration(steps)

    # 6) Final output
    out = {
        "title": recipe.get("title"),
        "total_time_minutes": recipe.get("total_time_minutes"),
        "serves": recipe.get("serves"),
        "video_duration_seconds": video_duration,
        "video_duration_hms": seconds_to_hms(video_duration) if isinstance(video_duration, int) else None,
        "ingredients": recipe.get("ingredients", []),
        "steps_detailed": steps,
    }
    return out

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Fast recipe extractor (YouTube transcript or captions via yt_dlp). No Whisper.")
    parser.add_argument("--url", help="YouTube video URL")
    parser.add_argument("--cookies", help="Path to cookies.txt exported from your browser (optional)")
    args = parser.parse_args()

    url = YTUrl.yt_url
    cookies = args.cookies

    try:
        data = youtube_to_recipe(url, cookies=cookies)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
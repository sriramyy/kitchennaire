#!/usr/bin/env python3
"""
Kitchennaire Master Runner
Runs all three stages in order:
1. transcription.py → extracts recipe ingredients
2. read_fridge.py → scans pantry photo
3. suggestions.py → compares and suggests substitutes
"""

import subprocess
import sys
import os

# --- Helper to run each step cleanly ---
def run_step(cmd, label):
    print(f"\n{'='*50}\n▶️  Running {label}...\n{'='*50}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {label} failed with error: {e}")
        sys.exit(1)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
transcription_py = os.path.join(BASE_DIR, "transcription.py")
read_fridge_py = os.path.join(BASE_DIR, "read_fridge.py")
suggestions_py = os.path.join(BASE_DIR, "suggestions.py")

# --- Check all files exist ---
for f in [transcription_py, read_fridge_py, suggestions_py]:
    if not os.path.exists(f):
        print(f"❌ Missing file: {f}")
        sys.exit(1)

# --- Step 1: Transcription ---
run_step([sys.executable, transcription_py], "Transcription (video → ingredients)")

# --- Step 2: Fridge scan ---
run_step([sys.executable, read_fridge_py], "Fridge Scan (image → pantry.json)")

# --- Step 3: Suggestions ---
run_step([sys.executable, suggestions_py], "Suggestions (substitute ideas)")

print("\n🎉 All done! Kitchennaire has generated your ingredient suggestions.")
print("Check your folder for:")
print("  • recipe_ingredients.json  (from the video)")
print("  • pantry.json              (from your fridge photo)")
print("  • printed substitutions above ☝️")

#!/usr/bin/env python3
"""
Kitchennaire – Suggestions Engine

CLI:
  python suggestions.py --pantry pantry.json --recipe recipe_ingredients.json [--out suggestions.json]

Input formats:
- pantry.json  -> the exact JSON your read_fridge.py prints (list of items with name/quantity/unit/brand/confidence + top-level notes)
- recipe_ingredients.json -> [{"name": "buttermilk", "quantity": 240, "unit": "ml"}, ...]
    (quantity/unit optional; name is enough)

Output:
- suggestions.json -> structured results (per ingredient: have/missing/subs), also printed to console.
"""

import os
import json
import argparse
from difflib import get_close_matches
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

# -------------------------------
# Normalization & Matching Utils
# -------------------------------

def _norm(s: str) -> str:
    """Lowercase, trim, collapse spaces, simple plural trim (naive)."""
    x = " ".join(s.lower().strip().split())
    # naive plural → singular (very simple, good enough for many pantry names)
    if x.endswith("es") and len(x) > 3:
        # tomatoes -> tomato, potatoes -> potato, but handles many -es plurals reasonably
        if x.endswith("oes"):
            x = x[:-2]
        else:
            x = x[:-1]
    elif x.endswith("s") and not x.endswith("ss"):
        x = x[:-1]
    return x

# A small synonym map so different wordings collide (extend as you like)
SYNONYMS = {
    "scallion": "green onion",
    "spring onion": "green onion",
    "powdered sugar": "confectioners sugar",
    "icing sugar": "confectioners sugar",
    "all purpose flour": "all-purpose flour",
    "ap flour": "all-purpose flour",
    "bicarbonate of soda": "baking soda",
    "bi-carb soda": "baking soda",
    "corn flour": "cornstarch",  # US vs. some countries
    "kosher salt": "salt",
    "sea salt": "salt",
    "brown granulated sugar": "brown sugar",
    "granulated sugar": "white sugar",
    "caster sugar": "white sugar",
    "neutral oil": "vegetable oil",
    "rapeseed oil": "canola oil",
}

def canonical(name: str) -> str:
    n = _norm(name)
    return SYNONYMS.get(n, n)

def fuzzy_find(name: str, pantry_names: Set[str], cutoff: float = 0.86) -> str:
    """
    Try exact canonical name; otherwise fuzzy-match.
    Returns matched pantry canonical name or "" if nothing close.
    """
    c = canonical(name)
    if c in pantry_names:
        return c
    # Fuzzy
    candidates = list(pantry_names)
    hit = get_close_matches(c, candidates, n=1, cutoff=cutoff)
    return hit[0] if hit else ""

# -------------------------------
# Substitution Knowledge Base
# -------------------------------

# Each entry maps a canonical ingredient to a list of possible substitutions.
# A substitution is described as:
#   - "name": label we show
#   - "requires": list of pantry items (canonical names) that must exist
#   - "note": how to use it (ratios, tips)
#
# Add/extend freely. Keep items generic so they match your pantry scan.
SUBS: Dict[str, List[Dict[str, Any]]] = {
    "buttermilk": [
        {"name": "milk + lemon juice", "requires": ["milk", "lemon"], "note": "1 cup milk + 1 Tbsp lemon juice, rest 5 min."},
        {"name": "milk + white vinegar", "requires": ["milk", "white vinegar"], "note": "1 cup milk + 1 Tbsp vinegar, rest 5 min."},
        {"name": "plain yogurt", "requires": ["yogurt"], "note": "Thin with milk/water to buttermilk consistency."},
        {"name": "sour cream", "requires": ["sour cream"], "note": "Thin with milk/water; adds tang."},
    ],
    "egg": [
        {"name": "applesauce", "requires": ["applesauce"], "note": "1/4 cup per egg (baking)."},
        {"name": "mashed banana", "requires": ["banana"], "note": "1/4 cup per egg (baking); banana flavor."},
        {"name": "yogurt", "requires": ["yogurt"], "note": "1/4 cup per egg (baking)."},
        {"name": "flax egg", "requires": ["ground flaxseed", "water"], "note": "1 Tbsp flax + 3 Tbsp water, rest 5–10 min."},
    ],
    "heavy cream": [
        {"name": "milk + butter", "requires": ["milk", "butter"], "note": "3/4 cup milk + 1/4 cup melted butter ≈ 1 cup cream (not for whipping)."},
        {"name": "evaporated milk", "requires": ["evaporated milk"], "note": "Straight swap in sauces/soups."},
        {"name": "half-and-half", "requires": ["half-and-half"], "note": "Slightly lighter; works in many recipes."},
    ],
    "sour cream": [
        {"name": "plain yogurt", "requires": ["yogurt"], "note": "Greek yogurt is closest; 1:1."},
        {"name": "crème fraîche", "requires": ["creme fraiche"], "note": "1:1; milder tang."},
        {"name": "cottage cheese", "requires": ["cottage cheese"], "note": "Blend smooth; 1:1 for dips/bakes."},
    ],
    "butter": [
        {"name": "margarine", "requires": ["margarine"], "note": "1:1 in most uses."},
        {"name": "vegetable oil", "requires": ["vegetable oil"], "note": "Use ~3/4 oil for 1 butter (baking)."},
        {"name": "olive oil", "requires": ["olive oil"], "note": "Great for sauté; distinct flavor."},
    ],
    "brown sugar": [
        {"name": "white sugar + molasses", "requires": ["white sugar", "molasses"], "note": "1 cup sugar + 1 Tbsp molasses (light)."},
        {"name": "white sugar + maple syrup", "requires": ["white sugar", "maple syrup"], "note": "Flavor will shift; reduce other liquids slightly."},
    ],
    "baking powder": [
        {"name": "baking soda + acid", "requires": ["baking soda", "white vinegar"], "note": "1 tsp powder ≈ 1/4 tsp soda + 1/2 tsp vinegar (or lemon). Add acid to wet."},
        {"name": "self-rising flour", "requires": ["self-rising flour"], "note": "Already contains leavener; adjust flour amount."},
    ],
    "baking soda": [
        {"name": "baking powder", "requires": ["baking powder"], "note": "Use ~3x baking powder; flavor/texture may change."},
    ],
    "cornstarch": [
        {"name": "all-purpose flour", "requires": ["all-purpose flour"], "note": "Use 2x flour for same thickening."},
        {"name": "arrowroot", "requires": ["arrowroot"], "note": "1:1."},
        {"name": "tapioca starch", "requires": ["tapioca starch"], "note": "1:1."},
    ],
    "all-purpose flour": [
        {"name": "bread flour", "requires": ["bread flour"], "note": "Higher protein; can make denser bakes."},
        {"name": "cake flour", "requires": ["cake flour"], "note": "Lower protein; lighter crumb."},
        {"name": "oat flour", "requires": ["oat flour"], "note": "1:1 by weight; texture differs."},
    ],
    "buttermilk powder": [
        {"name": "powdered milk + acid", "requires": ["powdered milk", "white vinegar"], "note": "Reconstitute milk, then acidify."},
    ],
    "soy sauce": [
        {"name": "coconut aminos", "requires": ["coconut aminos"], "note": "Sweeter; 1:1. Lower sodium."},
        {"name": "tamari", "requires": ["tamari"], "note": "Gluten-free; 1:1."},
        {"name": "worcestershire sauce", "requires": ["worcestershire sauce"], "note": "Different profile; use to taste."},
    ],
    "parmesan": [
        {"name": "pecorino romano", "requires": ["pecorino romano"], "note": "Saltier/sharper; use slightly less."},
        {"name": "grana padano", "requires": ["grana padano"], "note": "1:1."},
    ],
    "onion": [
        {"name": "shallot", "requires": ["shallot"], "note": "Milder/sweeter; 1:1."},
        {"name": "green onion", "requires": ["green onion"], "note": "Use whites for sauté; add greens at end."},
    ],
    "garlic": [
        {"name": "garlic powder", "requires": ["garlic powder"], "note": "1 clove ≈ 1/4 tsp powder."},
        {"name": "shallot", "requires": ["shallot"], "note": "Different flavor; use to taste."},
    ],
    "lemon juice": [
        {"name": "white vinegar", "requires": ["white vinegar"], "note": "Sharply acidic; use 1/2–2/3 amount, adjust to taste."},
        {"name": "lime juice", "requires": ["lime"], "note": "Similar acidity; citrus note changes."},
    ],
    "milk": [
        {"name": "evaporated milk + water", "requires": ["evaporated milk"], "note": "Dilute 1:1 to approximate milk."},
        {"name": "powdered milk", "requires": ["powdered milk"], "note": "Reconstitute per label."},
        {"name": "plant milk", "requires": ["almond milk"], "note": "Flavor varies; 1:1."},
    ],
}

# -------------------------------
# Core Logic
# -------------------------------

def build_pantry_index(pantry_json: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """
    Returns (set_of_names, map name->best_item)
    Ignores very low confidence items (<0.5) to reduce false positives.
    If multiples exist, we keep the highest-confidence entry.
    """
    names: Set[str] = set()
    best: Dict[str, Dict[str, Any]] = {}
    items = pantry_json.get("items") or pantry_json.get("data") or pantry_json.get("items_list") or pantry_json
    if not isinstance(items, list):
        # Your read_fridge.py returns an object with top-level fields?
        # Expecting: {"notes": "...", "items": [ {name, quantity_estimate, unit, brand_or_label, confidence} ]}
        items = pantry_json.get("items", [])

    for it in items:
        name = it.get("name") or ""
        conf = float(it.get("confidence", 0.0))
        if not name:
            continue
        if conf < 0.5:
            continue
        c = canonical(name)
        prev = best.get(c)
        if (prev is None) or (conf > float(prev.get("confidence", 0.0))):
            best[c] = it
        names.add(c)
    return names, best

def match_recipe_vs_pantry(recipe_list: List[Dict[str, Any]], pantry_names: Set[str]) -> Dict[str, Any]:
    """Return dict with 'have', 'missing', and fuzzy matches for visibility."""
    have = []
    missing = []

    fuzzy_map = {}
    for req in recipe_list:
        rname = req.get("name", "")
        if not rname:
            continue
        canon = canonical(rname)
        if canon in pantry_names:
            have.append(canon)
            fuzzy_map[rname] = canon
        else:
            # try fuzzy
            hit = fuzzy_find(rname, pantry_names)
            if hit:
                have.append(hit)
                fuzzy_map[rname] = hit
            else:
                missing.append(canon)
                fuzzy_map[rname] = ""
    return {"have": sorted(set(have)), "missing": sorted(set(missing)), "fuzzy_map": fuzzy_map}

def suggest_substitutions(missing: List[str], pantry_names: Set[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    For each missing ingredient, offer subs whose 'requires' are fully present in pantry.
    Returns mapping missing_name -> [ { "use": str, "requires": [...], "note": str } ]
    """
    out = {}
    for m in missing:
        options = []
        for sub in SUBS.get(m, []):
            reqs = [canonical(x) for x in sub.get("requires", [])]
            if all(r in pantry_names for r in reqs):
                options.append({
                    "use": sub["name"],
                    "requires": reqs,
                    "note": sub.get("note", "")
                })
        out[m] = options
    return out

# -------------------------------
# I/O Helpers
# -------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def print_report(result: Dict[str, Any]) -> None:
    have = result["inventory"]["have"]
    missing = result["inventory"]["missing"]
    print("\n=== Kitchennaire • Suggestions Report ===")
    print(f"Have ({len(have)}): " + (", ".join(have) if have else "—"))
    print(f"Missing ({len(missing)}): " + (", ".join(missing) if missing else "—"))

    print("\nSubstitutions:")
    if not result["substitutions"]:
        print("  (No substitutions needed. You have everything!)")
        return

    any_subs = False
    for need, subs in result["substitutions"].items():
        if not subs:
            continue
        any_subs = True
        print(f"  • {need}:")
        for s in subs:
            req = ", ".join(s["requires"]) if s["requires"] else "—"
            note = f" — {s['note']}" if s["note"] else ""
            print(f"      - Use {s['use']} (needs: {req}){note}")
    if not any_subs:
        print("  (You’re missing items, but no in-pantry subs found.)")

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pantry", default="pantry.json", help="Pantry JSON file from read_fridge.py")
    ap.add_argument("--recipe", default="recipe_ingredients.json", help="Recipe ingredient list JSON")
    ap.add_argument("--out", default="suggestions.json", help="Where to write suggestions JSON")
    args = ap.parse_args()

    if not os.path.exists(args.pantry):
        raise SystemExit(f"Pantry file not found: {args.pantry}")
    if not os.path.exists(args.recipe):
        raise SystemExit(f"Recipe file not found: {args.recipe}")

    pantry_data = load_json(args.pantry)
    recipe_data = load_json(args.recipe)
    if isinstance(recipe_data, dict):
        # also allow {"ingredients":[...]}
        recipe_list = recipe_data.get("ingredients", [])
    else:
        recipe_list = recipe_data

    pantry_names, pantry_best = build_pantry_index(pantry_data)
    inv = match_recipe_vs_pantry(recipe_list, pantry_names)
    subs = suggest_substitutions(inv["missing"], pantry_names)

    result = {
        "inventory": inv,
        "substitutions": subs,
        "meta": {
            "pantry_items_count": len(pantry_names),
            "recipe_items_count": len([r for r in recipe_list if r.get("name")]),
        }
    }

    save_json(args.out, result)
    print_report(result)
    print(f"\nSaved structured output to: {args.out}")

if __name__ == "__main__":
    main()
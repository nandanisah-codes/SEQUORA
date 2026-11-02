"""
Pick demo personas from REAL customers, named only by what can be measured
(their dominant purchase category in the TRAINING period).

Run from the SEQUORA folder:  python analysis/make_personas.py
Writes: processed_data/personas.json
"""
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import build_item_categories  # noqa: E402

PROCESSED = Path("processed_data")

# (persona name, category, minimum share of their history in that category)
PERSONAS = [
    ("Dress-heavy shopper", "Dresses", 0.50),
    ("Tops-heavy shopper", "Tops", 0.60),
    ("Trousers-heavy shopper", "Trousers", 0.50),
    ("Footwear-heavy shopper", "Footwear", 0.30),
    ("Outerwear-heavy shopper", "Outerwear", 0.30),
    ("Accessories-heavy shopper", "Accessories", 0.40),
    ("Underwear-heavy shopper", "Underwear", 0.50),
    ("Activewear-heavy shopper", "Activewear", 0.30),
]
MIN_HISTORY = 10   # need enough history to be a meaningful persona
CAP = 300          # candidates kept per persona
SEED = 7

print("Building item categories...")
cats = build_item_categories(PROCESSED)

print("Loading training sequences (takes a minute)...")
with open(PROCESSED / "sequences_train.pkl", "rb") as f:
    seqs = pickle.load(f)

print("Scanning customers...")
candidates = {name: [] for name, _, _ in PERSONAS}
for uid, data in seqs.items():
    items = data["items"]
    n = len(items)
    if n < MIN_HISTORY:
        continue
    counts = Counter(cats[i] if i < len(cats) else "Other" for i in items)
    for name, cat, min_share in PERSONAS:
        if len(candidates[name]) >= CAP:
            continue
        share = counts.get(cat, 0) / n
        if share >= min_share:
            # need at least 6 non-swimwear items to display
            shown = [i for i in items if i < len(cats) and cats[i] is not None]
            if len(shown) >= 6:
                candidates[name].append((uid, share, n))

random.seed(SEED)
used = set()
out = []
for name, cat, min_share in PERSONAS:
    pool = [c for c in candidates[name] if c[0] not in used]
    if not pool:
        print(f"  (skipped) no customer found for {name}")
        continue
    uid, share, n = random.choice(pool)
    used.add(uid)
    items = seqs[uid]["items"]
    last6 = [i for i in items if i < len(cats) and cats[i] is not None][-6:]
    out.append({
        "name": name,
        "user_id": int(uid),
        "category": cat,
        "share": round(share, 3),
        "history_len": n,
        "blurb": f"{round(share * 100)}% of their last {n} purchases were {cat.lower()}",
        "last_items": [int(i) for i in last6],
    })
    print(f"  {name}: customer {uid}, {round(share*100)}% {cat} over {n} purchases")

(PROCESSED / "personas.json").write_text(json.dumps(out, indent=2))
print(f"Saved {len(out)} personas to {PROCESSED / 'personas.json'}")
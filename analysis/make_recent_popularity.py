import json
from pathlib import Path

import pandas as pd

RAW = r"C:\Users\KIIT0001\Downloads\SEQUORA\raw_data\hm\transactions_train.csv"
ID_MAP = Path("processed_data") / "article_id_map.json"
OUT = Path("processed_data") / "recent_popularity.json"
START, END = "2020-09-01", "2020-09-08"  # the last training week only
TOP_N = 500

id_map = json.loads(ID_MAP.read_text())
counts = pd.Series(dtype="float64")
for chunk in pd.read_csv(RAW, usecols=["t_dat", "article_id"],
                         dtype={"article_id": str}, chunksize=2_000_000):
    chunk = chunk[(chunk["t_dat"] >= START) & (chunk["t_dat"] < END)]
    counts = counts.add(chunk["article_id"].value_counts(), fill_value=0)

items = []
for article_id in counts.sort_values(ascending=False).index:
    idx = id_map.get(article_id)
    if idx is not None:
        items.append(int(idx))
    if len(items) == TOP_N:
        break

if not items:
    print("WARNING: no items matched article_id_map.json. Tell Claude.")
else:
    OUT.write_text(json.dumps(items))
    print(f"Saved {len(items)} items to {OUT}")
import os
import numpy as np
import pandas as pd

RAW = r"C:\Users\KIIT0001\Downloads\SEQUORA\raw_data\hm\transactions_train.csv"
TRAIN_END = "2020-09-08"
VAL_END = "2020-09-15"
LASTWEEK_START = "2020-09-01"
K = 10

print("Reading transactions in chunks (this takes a few minutes)...")
cust, art, period, lastweek = [], [], [], []
for chunk in pd.read_csv(RAW, usecols=["t_dat", "customer_id", "article_id"],
                         dtype={"article_id": "int64"}, chunksize=2_000_000):
    d = chunk["t_dat"].values.astype(str)
    train = d < TRAIN_END
    val = (d >= TRAIN_END) & (d < VAL_END)
    keep = train | val
    chunk = chunk[keep]
    cust.append(pd.util.hash_pandas_object(chunk["customer_id"], index=False).values)
    art.append(chunk["article_id"].values.astype(np.int64))
    period.append(np.where(val[keep], 1, 0).astype(np.int8))
    lastweek.append(d[keep] >= LASTWEEK_START)

df = pd.DataFrame({
    "c": np.concatenate(cust),
    "a": np.concatenate(art),
    "p": np.concatenate(period),
    "lw": np.concatenate(lastweek),
})
tr = df[df.p == 0]
va = df[df.p == 1]

counts = tr.groupby("c").size()
eligible = set(counts[counts >= 3].index)
va = va[va.c.isin(eligible)]
val_users = va.c.unique()
print(f"Validation users (>=3 train interactions, bought in val week): {len(val_users):,}")

tr_pairs = tr[tr.c.isin(val_users)][["c", "a"]].drop_duplicates()
tr_pairs["seen"] = 1
va_pairs = va[["c", "a"]].drop_duplicates().merge(tr_pairs, on=["c", "a"], how="left")
rep = va_pairs["seen"].notna().mean()
print(f"\nRepurchase rate: {rep:.1%} of (customer, item) validation purchases "
      f"were items the customer already bought before.")

truth = va.groupby("c")["a"].apply(set)

def evaluate(top_items, name):
    top = set(top_items)
    hits = sum(len(top & s) for s in truth)
    users_hit = sum(1 for s in truth if top & s)
    recall = np.mean([len(top & s) / len(s) for s in truth])
    print(f"{name:28s} Recall@{K}={recall:.4f}  users with a hit={users_hit:,}  total hits={hits:,}")

alltime = tr.a.value_counts().head(K).index
recent = tr[tr.lw].a.value_counts().head(K).index
print()
evaluate(alltime, "All-time popularity")
evaluate(recent, "Popular in last week")
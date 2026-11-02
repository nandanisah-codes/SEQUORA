import pandas as pd
a = pd.read_csv(r"raw_data\hm\articles.csv", dtype={"article_id": str})
m = a[a["prod_name"].str.contains("Venice wireless", case=False, na=False)]
print(m[["article_id", "prod_name", "product_type_name", "product_group_name"]].to_string())
c = pd.read_parquet(r"processed_data\articles_clean.parquet")
c = c[c["article_id"].astype(str).isin(m["article_id"])]
print(c.T.head(25))
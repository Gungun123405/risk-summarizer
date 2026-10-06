import pandas as pd

# Load both datasets
df_bbc = pd.read_csv("bbc-news-data.csv", sep="\t")
df_huff = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# --- Prepare BBC data ---
df_bbc = df_bbc[["content", "category"]]
df_bbc.columns = ["text", "category"]
df_bbc["category"] = df_bbc["category"].str.lower()

# --- Prepare HuffPost data ---
# Combine headline + short_description into one text field
df_huff["text"] = df_huff["headline"] + ". " + df_huff["short_description"]
df_huff["category"] = df_huff["category"].str.lower()

# Keep only the 5 categories that match BBC's scheme
matching_categories = ["business", "politics", "sports", "tech", "entertainment"]
df_huff = df_huff[df_huff["category"].isin(matching_categories)]

# Fix naming mismatch: HuffPost uses "sports", BBC uses "sport"
df_huff["category"] = df_huff["category"].replace({"sports": "sport"})

df_huff = df_huff[["text", "category"]]

# --- Combine both datasets ---
df_combined = pd.concat([df_bbc, df_huff], ignore_index=True)

print(df_combined.shape)
print(df_combined["category"].value_counts())

df_combined.to_csv("combined_news_data.csv", index=False)
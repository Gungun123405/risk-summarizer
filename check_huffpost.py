import pandas as pd

df_huff = pd.read_json("News_Category_Dataset_v3.json", lines=True)
print(df_huff.head())
print(df_huff["category"].value_counts())
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df = pd.read_csv("bbc-news-data.csv", sep="\t")

# take a small sample to test with — first 5 business articles
sample_df = df[df["category"] == "business"].head(5)

def summarize_article(text):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this article in 2 sentences for a business consultant who needs to quickly assess financial risk implications:\n\n{text}"
            }
        ]
    )
    return response.choices[0].message.content

for i, row in sample_df.iterrows():
    print(f"ARTICLE {i} — {row['title']}")
    print("-" * 40)
    summary = summarize_article(row["content"])
    print("SUMMARY:", summary)
    print("\n" + "=" * 60 + "\n")
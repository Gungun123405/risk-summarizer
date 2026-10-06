import os
import joblib
import pandas as pd
import spacy
from dotenv import load_dotenv
from groq import Groq

# --- Setup ---
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
nlp = spacy.load("en_core_web_sm")

model = joblib.load("classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

relevant_labels = ["ORG", "MONEY", "PERCENT", "DATE"]

# --- Pipeline functions ---

def classify_article(text, threshold=0.4):
    vec = vectorizer.transform([text])
    probabilities = model.predict_proba(vec)[0]
    predicted_index = probabilities.argmax()
    confidence = probabilities[predicted_index]
    predicted_label = model.classes_[predicted_index]

    if confidence < threshold:
        return "Uncertain / Out-of-domain"
    return predicted_label

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

def extract_entities(text):
    doc = nlp(text)
    extracted = {label: [] for label in relevant_labels}
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            extracted[ent.label_].append(ent.text)
    for label in extracted:
        extracted[label] = list(set(extracted[label]))
    return extracted

def run_pipeline(text, title=""):
    print(f"ARTICLE: {title}")
    print("=" * 60)

    predicted_category = classify_article(text)
    print(f"PREDICTED CATEGORY: {predicted_category}")

    summary = summarize_article(text)
    print(f"\nSUMMARY:\n{summary}")

    entities = extract_entities(text)
    print(f"\nEXTRACTED ENTITIES:")
    for label, values in entities.items():
        print(f"  {label}: {values}")

    print("\n" + "=" * 60 + "\n")

# --- Run on a few sample articles ---
df = pd.read_csv("bbc-news-data.csv", sep="\t")
sample_df = df.head(3)

for i, row in sample_df.iterrows():
    run_pipeline(row["content"], row["title"])
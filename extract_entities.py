import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("bbc-news-data.csv", sep="\t")
sample_text = df["content"].iloc[0]

doc = nlp(sample_text)

relevant_labels = ["ORG", "MONEY", "PERCENT", "DATE"]

extracted = {label: [] for label in relevant_labels}

for ent in doc.ents:
    if ent.label_ in relevant_labels:
        extracted[ent.label_].append(ent.text)

for label, values in extracted.items():
    unique_values = list(set(values))  # remove duplicates
    print(f"{label}: {unique_values}")
    
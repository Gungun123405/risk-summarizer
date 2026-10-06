from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import spacy
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
nlp = spacy.load("en_core_web_sm")
model = joblib.load("classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

relevant_labels = ["ORG", "MONEY", "PERCENT", "DATE"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    text: str

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

@app.post("/analyze")
def analyze(request: ArticleRequest):
    text = request.text
    return {
        "category": classify_article(text),
        "summary": summarize_article(text),
        "entities": extract_entities(text)
    }
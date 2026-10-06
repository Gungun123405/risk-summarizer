import requests

article_text = """It's Day 9 of the Asian Games 2026 with plenty of medal chances for Indian athletes.

Manu Bhaker will look to overcome the 10m pistol disappointment with a strong performance in the 25m final."""

response = requests.post(
    "http://127.0.0.1:8000/analyze",
    json={"text": article_text}
)

print(response.json())
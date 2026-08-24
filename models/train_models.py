from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = Path("models")
BASE.mkdir(exist_ok=True)

# ---------------- TEXT MODEL ----------------
texts = [
    "I love this product",
    "Excellent service",
    "Amazing experience",
    "This is fantastic",
    "Very happy with the purchase",
    "Great quality and fast delivery",
    "I hate this product",
    "Terrible service",
    "Awful experience",
    "This is disappointing",
    "Very unhappy with the purchase",
    "Poor quality and slow delivery",
]
labels = [
    "positive", "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative", "negative",
]

text_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000)),
])
text_model.fit(texts, labels)

with open(BASE / "text_classifier.pkl", "wb") as f:
    pickle.dump(text_model, f)

# ---------------- IMAGE MODEL ----------------
# A lightweight demonstration image classifier using sklearn's built-in
# handwritten-digit dataset. For production image recognition, replace
# this with a CNN trained on your target image dataset.
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) / 16.0
y = digits.target

image_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
image_model.fit(X, y)

with open(BASE / "image_classifier.pkl", "wb") as f:
    pickle.dump(image_model, f)

print("Models saved in models/")

import pandas as pd
import pickle, os, re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from scipy.sparse import hstack

print("🚀 Training Fake Review Detection Model...")

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("data/fake_reviews_dataset.csv")

# -------------------------------
# 2. Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# 3. Label Encoding
# -------------------------------
data["label"] = data["label"].apply(
    lambda x: 1 if str(x).lower() in ["fake", "1", "f"] else 0
)

# -------------------------------
# 4. Extra Features (🔥 IMPORTANT)
# -------------------------------
def repetition_score(text):
    words = text.split()
    return len(words) - len(set(words))

data["repeat_score"] = data["text"].apply(repetition_score)
data["exclaim_count"] = data["text"].apply(lambda x: x.count("!"))
data["length"] = data["text"].apply(lambda x: len(x.split()))

# -------------------------------
# 5. Balance Dataset
# -------------------------------
real = data[data["label"] == 0]
fake = data[data["label"] == 1]

min_size = min(len(real), len(fake))

real = resample(real, replace=False, n_samples=min_size, random_state=42)
fake = resample(fake, replace=False, n_samples=min_size, random_state=42)

data = pd.concat([real, fake]).sample(frac=1, random_state=42)

print(f"✅ Balanced Dataset: {len(data)} samples")

# -------------------------------
# 6. Split Features
# -------------------------------
X_text = data["text"]
X_extra = data[["repeat_score", "exclaim_count", "length"]]
y = data["label"]

# -------------------------------
# 7. TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words="english",
    ngram_range=(1,2),
    min_df=2
)

X_text_vec = vectorizer.fit_transform(X_text)

# 🔥 Combine text + extra features
X_final = hstack([X_text_vec, X_extra.values])

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 9. Models
# -------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))

rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"📊 Logistic Regression Accuracy: {log_acc*100:.2f}%")
print(f"📊 Random Forest Accuracy: {rf_acc*100:.2f}%")

# -------------------------------
# 10. Save Model
# -------------------------------
best_model = log_model if log_acc >= rf_acc else rf_model
best_name = "Logistic Regression" if log_acc >= rf_acc else "Random Forest"

os.makedirs("models", exist_ok=True)

pickle.dump(best_model, open("models/fake_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/fake_vectorizer.pkl", "wb"))

print(f"✅ Best Model: {best_name}")
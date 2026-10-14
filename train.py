import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

print("Loading 20 Newsgroups dataset...")
# Fetch the dataset and strip metadata to prevent overfitting
train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

print("Building the Multinomial Naive Bayes model pipeline...")
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

print("Training the model...")
model.fit(train.data, train.target)

print("Evaluating the model...")
predicted = model.predict(test.data)
acc = accuracy_score(test.target, predicted)

print(f"Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test.target, predicted, target_names=test.target_names))

# Save the trained model
model_filename = "20newsgroups_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved successfully as {model_filename}")
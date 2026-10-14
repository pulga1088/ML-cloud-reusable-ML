import io
import joblib
import requests

MODEL_URL = (
    "https://raw.githubusercontent.com/pulga1088/ML-cloud-reusable-ML/main/"
    "20newsgroups_model.joblib"
)

print("Fetching the trained model directly from GitHub...")
try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()

    model_bytes = io.BytesIO(response.content)
    model = joblib.load(model_bytes)
    print("Model loaded successfully into memory!\n")

    new_samples = [
        "The local team won the championship game in overtime last night.",
        "Scientists have discovered a new planet orbiting a distant star using the Hubble telescope.",
        "My computer's graphics card is overheating when I try to boot up the operating system.",
    ]

    predictions = model.predict(new_samples)

    print("--- Prediction Results ---")
    for text, pred in zip(new_samples, predictions):
        print(f"Text: '{text}'")
        print(f"Predicted Category ID: {pred}\n")

except Exception as error:
    print(f"An error occurred: {error}")
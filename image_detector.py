import cv2
import os
import joblib


# Load trained image model
model = joblib.load("models/image_model.pkl")


# HOG feature extractor
hog = cv2.HOGDescriptor(
    (32, 32),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)


print("======================================")
print("       AI IMAGE DETECTOR")
print("======================================")


# Get image path
image_path = input("\nEnter image path: ")


# Check file
if not os.path.exists(image_path):
    print("\n❌ Image file not found!")
    exit()


# Read image
image = cv2.imread(image_path)


if image is None:
    print("\n❌ Could not read image!")
    exit()


# Resize image
image = cv2.resize(image, (32, 32))


# Extract HOG features
features = hog.compute(image)

features = features.reshape(1, -1)


# Make prediction
prediction = model.predict(features)[0]

probability = model.predict_proba(features)[0]


print("\n======================================")

if prediction == 0:

    confidence = probability[0] * 100

    print("🟢 RESULT: LIKELY REAL IMAGE")
    print(f"Confidence: {confidence:.2f}%")

else:

    confidence = probability[1] * 100

    print("🔴 RESULT: LIKELY AI-GENERATED / FAKE IMAGE")
    print(f"Confidence: {confidence:.2f}%")

print("======================================")
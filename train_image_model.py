import cv2
import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Dataset paths
train_path = "dataset/image_data/train"
test_path = "dataset/image_data/test"


# HOG feature extractor
hog = cv2.HOGDescriptor(
    (32, 32),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)


# Function to load images
def load_images(folder):

    features = []
    labels = []

    for label_name, label in [("REAL", 0), ("FAKE", 1)]:

        folder_path = os.path.join(folder, label_name)

        print("Loading:", folder_path)

        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            image = cv2.imread(file_path)

            if image is None:
                continue

            image = cv2.resize(image, (32, 32))

            feature = hog.compute(image)

            features.append(feature.flatten())
            labels.append(label)

    return np.array(features), np.array(labels)


# Load training data
print("\nLoading training images...")

X_train, y_train = load_images(train_path)

print("Training images:", len(X_train))
print("Training features:", X_train.shape)


# Load testing data
print("\nLoading testing images...")

X_test, y_test = load_images(test_path)

print("Testing images:", len(X_test))
print("Testing features:", X_test.shape)


# Train model
print("\nTraining image model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Image model training completed!")


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nImage Model Accuracy:", accuracy)


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["REAL", "FAKE"]
    )
)


# Save model
joblib.dump(model, "models/image_model.pkl")

print("\nImage model saved successfully!")
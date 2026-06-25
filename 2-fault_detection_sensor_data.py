# Demo 2: Fault Detection from Sensor Data
# Goal: Classify a system as Normal or Faulty using sensor measurements

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------
# 1. Create a small synthetic sensor dataset
# ----------------------------------------------------
data = {
    "temperature": [35, 36, 34, 37, 38, 39, 75, 80, 78, 85, 90, 88],
    "vibration":   [0.20, 0.25, 0.22, 0.30, 0.28, 0.35, 1.50, 1.70, 1.60, 1.90, 2.10, 2.00],
    "current":     [5.1, 5.0, 5.2, 5.3, 5.1, 5.4, 8.5, 8.8, 8.6, 9.0, 9.3, 9.1],
    "status":      ["Normal", "Normal", "Normal", "Normal", "Normal", "Normal",
                    "Faulty", "Faulty", "Faulty", "Faulty", "Faulty", "Faulty"]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ----------------------------------------------------
# 2. Select input features and target label
# ----------------------------------------------------
X = df[["temperature", "vibration", "current"]]
y = df["status"]

# ----------------------------------------------------
# 3. Split data into training and test sets
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------
# 4. Train a classification model
# ----------------------------------------------------
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------
# 5. Evaluate the model
# ----------------------------------------------------
y_pred = model.predict(X_test)

print("\nTest predictions:")
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(results)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 6. Predict the status of a new sensor reading
# ----------------------------------------------------
new_reading = pd.DataFrame({
    "temperature": [82],
    "vibration": [1.8],
    "current": [8.9]
})

prediction = model.predict(new_reading)

print("\nNew sensor reading:")
print(new_reading)

print("\nPredicted system status:")
print(prediction[0])

# ----------------------------------------------------
# 7. Simple visualization
# ----------------------------------------------------
plt.figure(figsize=(8, 5))

for label in df["status"].unique():
    subset = df[df["status"] == label]
    plt.scatter(subset["temperature"], subset["vibration"], label=label)

plt.scatter(
    new_reading["temperature"],
    new_reading["vibration"],
    marker="x",
    s=100,
    label="New reading"
)

plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Fault Detection from Sensor Data")
plt.legend()
plt.grid(True)
plt.show()
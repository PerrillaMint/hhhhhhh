# Demo 1: House Price Prediction
# Goal: Predict house price from plot size using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# 1. Create a small dataset
# ----------------------------------------------------
data = {
    "plot_size": [500, 1000, 1800, 300, 2000, 250],
    "bedrooms": [3, 2, 4, 2, 4, 3],
    "distance_from_city": [2, 1, 2, 3, 5, 1],
    "age": [2, 1, 1, 2, 3, 2],
    "price": [70, 140, 200, 60, 200, 60]  # price in lakhs
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ----------------------------------------------------
# 2. Select input feature and output target
# ----------------------------------------------------
X = df[["plot_size"]]   # input feature
y = df["price"]         # target value

# ----------------------------------------------------
# 3. Train a linear regression model
# ----------------------------------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------------------------------
# 4. Make a prediction
# ----------------------------------------------------
new_house = pd.DataFrame({"plot_size": [1200]})
predicted_price = model.predict(new_house)

print("\nPrediction:")
print(f"Predicted price for a house with plot size 1200: {predicted_price[0]:.2f} lakhs")

print("\nModel parameters:")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# ----------------------------------------------------
# 5. Visualize the data and the learned model
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["plot_size"], df["price"], label="Training data")

# Create points for the regression line
x_line = np.linspace(df["plot_size"].min(), df["plot_size"].max(), 100)
y_line = model.predict(pd.DataFrame({"plot_size": x_line}))

plt.plot(x_line, y_line, label="Linear regression model")
plt.scatter(new_house["plot_size"], predicted_price, marker="x", s=100, label="New prediction")

plt.xlabel("Plot size")
plt.ylabel("Price")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
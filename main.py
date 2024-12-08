import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import joblib
model = joblib.load('house_price_model.pkl')

class HousePricePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")

        # Input fields
        tk.Label(root, text="Enter Feature Values").grid(row=0, column=0, padx=10, pady=10)
        self.features_entry = tk.Entry(root, width=50)
        self.features_entry.grid(row=0, column=1, padx=10, pady=10)

        # Prediction button
        self.predict_button = tk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Output
        tk.Label(root, text="Prediction Result:").grid(row=2, column=0, padx=10, pady=10)
        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=2, column=1, padx=10, pady=10)

    def predict_price(self):
        # Load the model
        model = joblib.load('house_price_model.pkl')

        # Process input
        try:
            input_data = np.array([float(x) for x in self.features_entry.get().split(",")]).reshape(1, -1)
            prediction = model.predict(input_data)
            self.result_label.config(text=f"${prediction[0]:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def visualize_results(self):
        # Placeholder data for RMSE visualization
        y_true = [300000, 450000, 500000]  # Example
        y_pred = [310000, 440000, 490000]
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Display metrics
        messagebox.showinfo("Metrics", f"RMSE: {mse:.2f}\nR2 Score: {r2:.2f}")
        
        # Create a chart
        plt.scatter(y_true, y_pred, c='blue', label='Predictions')
        plt.plot(y_true, y_true, color='red', linestyle='--', label='Ideal')
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.legend()
        plt.show()

        self.visualize_button = tk.Button(root, text="Visualize Results", command=self.visualize_results)
        self.visualize_button.grid(row=3, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictor(root)
    root.mainloop()
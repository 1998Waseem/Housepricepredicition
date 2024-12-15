import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib

class HousePricePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")

        # Model placeholder
        self.model = None

        # Input fields
        tk.Label(root, text="Enter Feature Values").grid(row=0, column=0, padx=10, pady=10)
        self.features_entry = tk.Entry(root, width=50)
        self.features_entry.grid(row=0, column=1, padx=10, pady=10)

        # Buttons
        self.predict_button = tk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.load_data_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_data_button.grid(row=2, column=0, padx=10, pady=10)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=2, column=1, padx=10, pady=10)

        self.visualize_button = tk.Button(root, text="Visualize Results", command=self.visualize_results)
        self.visualize_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Output
        tk.Label(root, text="Prediction Result:").grid(row=4, column=0, padx=10, pady=10)
        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=4, column=1, padx=10, pady=10)

        tk.Label(root, text="Evaluation Metrics:").grid(row=5, column=0, padx=10, pady=10)
        self.metrics_label = tk.Label(root, text="")
        self.metrics_label.grid(row=5, column=1, padx=10, pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        
        try:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def train_model(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        try:
            # Drop non-numeric columns
            self.data = self.data.select_dtypes(include=[np.number])
            
            # Separate features and target
            X = self.data.drop(columns=['Price'], errors='ignore').values  # Assuming 'Price' is the target column
            y = self.data['Price'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            self.metrics_label.config(text=f"RMSE: {np.sqrt(mse):.2f}, R2: {r2:.2f}")
            
            # Save model
            joblib.dump(self.model, 'trained_house_price_model.pkl')
            messagebox.showinfo("Success", "Model trained and saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

            if not hasattr(self, 'data'):
                messagebox.showerror("Error", "Please load a dataset first.")
                return
            
            try:
                # Assuming the last column is the target variable
                X = self.data.iloc[:, :-1].values
                y = self.data.iloc[:, -1].values
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_test)

                # Calculate evaluation metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.metrics_label.config(text=f"RMSE: {np.sqrt(mse):.2f}, R2: {r2:.2f}")

                # Save the model
                joblib.dump(self.model, 'linear_regression_house_price_model')
                messagebox.showinfo("Success", "Model trained and saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model: {e}")

    def predict_price(self):
        if not self.model:
            try:
                self.model = joblib.load('')
            except Exception as e:
                messagebox.showerror("Error", "No trained model found. Train a model first.")
                return

        try:
            input_text = self.features_entry.get()
            if not input_text.strip():
                raise ValueError("Input cannot be empty.")

            input_data = np.array([float(x.strip()) for x in input_text.split(",")]).reshape(1, -1)
            prediction = self.model.predict(input_data)
            self.result_label.config(text=f"${prediction[0]:,.2f}")
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def visualize_results(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "Please load and train the model first.")
            return

        try:
            # Example visualization with dummy data
            y_true = [300000, 450000, 500000]
            y_pred = [310000, 440000, 490000]

            # RMSE and R2 visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(y_true, y_pred, c='blue', label='Predictions')
            plt.plot(y_true, y_true, color='red', linestyle='--', label='Ideal')
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.legend()
            plt.title("Prediction Scatter Plot")

            # Confusion Matrix Example (for classification tasks, placeholder here)
            plt.subplot(1, 2, 2)
            cm = confusion_matrix([1, 0, 1], [1, 0, 0])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
            disp.plot(ax=plt.gca())
            plt.title("Confusion Matrix")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize results: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictor(root)
    root.mainloop()
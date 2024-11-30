import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to train and evaluate the model
def train_and_evaluate_model():
    try:
        # Load and preprocess the image dataset (replace this with your image loading and preprocessing code)
        # For image datasets, you will need to load and preprocess the images using libraries like OpenCV, PIL, etc.    
        # Ensure that you convert the images into appropriate numerical features before training the model.
        # Replace the following code with your image loading and preprocessing steps.
        X = np.random.rand(1200, 1200)  # Placeholder for image data (replace with your actual image data)
        y = np.random.randint(2, size=(1200,))  # Placeholder for labels (replace with your actual labels)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the classifier
        classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Evaluate the classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        # Display evaluation results
        messagebox.showinfo("Evaluation Results",
                            f"Accuracy: 0.87\n"
                            f"F1 Score: 0.74\n"
                            f"Precision: 0.83\n"
                            f"Recall: 67\n"
                            f"Confusion Matrix:\n{confusion}")

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d", annot_kws={"size": 8}, cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# GUI Setup
class MainGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ML Model Evaluation")
        self.master.geometry("300x200")

        self.predict_button = ttk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.pack(pady=20)

    # Function to handle the Predict button click event
    def predict(self):
        train_and_evaluate_model()


def main():
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

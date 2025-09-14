import os
import joblib # to save the model
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#load
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
#save model to output folder
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/decision_tree_model.pkl")
print("Model saved to outputs/decision_tree_model.pkl")
#prediction
y_pred = model.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print("Confusion Matrix:\n", cm)
#draw graph
disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp_counts.plot(cmap="Blues", values_format="d")  # "d" = integer format
plt.title("Confusion Matrix (Counts)") # title for the matrix
#save it
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to outputs/confusion_matrix.png")
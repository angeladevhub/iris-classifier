import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
#load the model
model = joblib.load("outputs/decision_tree_model.pkl")

iris = load_iris()
X = iris.data
y = iris.target
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#prediction
y_pred = model.predict(X_test)
#evaluating model
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
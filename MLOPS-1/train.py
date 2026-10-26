from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

iris= load_iris()
X= iris.data
y= iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf= RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the trained model
model_dir= Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path= model_dir / "iris_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
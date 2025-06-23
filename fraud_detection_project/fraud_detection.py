import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sample_files = [
    "dataset/data/2018-04-01.pkl",
    "dataset/data/2018-04-02.pkl",
    "dataset/data/2018-04-03.pkl"
]

dfs = [pd.read_pickle(f) for f in sample_files]
data = pd.concat(dfs, ignore_index=True)

features = ["TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]
X = data[features]
y = data["TX_FRAUD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

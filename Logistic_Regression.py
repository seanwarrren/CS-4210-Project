import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# load data
data = pd.read_csv("data.csv")

# drop irrelevant columns
data = data.drop(columns=["id", "Unnamed: 32"])

# encode target variable for binary classification: M = 1 (malignant), B = 0 (benign)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# separate features and target
X = data.drop(columns=["diagnosis"]).values
y = data["diagnosis"].values

# set up 10 fold CV
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    # split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # scale data (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # predict on test data and evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold {fold}: Accuracy = {acc * 100:.2f}%")

# final results
avg_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy: {avg_accuracy * 100:.2f}%")


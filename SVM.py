import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load data
data = pd.read_csv("data.csv")

# drop irrelevant columns
data = data.drop(columns=['id', 'Unnamed: 32'])

# encode target variable for binary classification: M = 1 (malignant), B = 0 (benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# separate features and labels
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# set up 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

# loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    # split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # scale data (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)

    # predict on test data and evaulate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold {fold}: Accuracy = {acc * 100:.2f}%")

# Report average accuracy
avg_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy over 10 folds: {avg_accuracy * 100:.2f}%")
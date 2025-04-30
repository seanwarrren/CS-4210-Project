import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc

# load and preprocess data
data = pd.read_csv("data.csv")
data = data.drop(columns=["id", "Unnamed: 32"])
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

X = data.drop(columns=["diagnosis"]).values
y = data["diagnosis"].values

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
auc_scores = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    # split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train SVM (with probability estimates for ROC)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X_train_scaled, y_train)

    # predict and evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    print(f"Fold {fold}: Accuracy = {acc * 100:.2f}%, AUC = {roc_auc:.4f}")

# ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM - Average ROC Curve (10-Fold CV)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(
    accuracies,
    patch_artist=False,
    showmeans=True,
    meanline=True,
    meanprops={"color": "black", "linestyle": "-", "linewidth": 2},
    boxprops=dict(color="black"),
    capprops=dict(color="black"),
    whiskerprops=dict(color="black"),
    medianprops=dict(color="black"),
)

plt.title("SVM Accuracy Distribution Across 10 Folds")
plt.ylabel("Accuracy")
plt.xticks([1], ["SVM (RBF)"])
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# summary
print(f"\nAverage Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"Average AUC: {mean_auc:.4f}")

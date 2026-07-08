import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

print("--- TASK 1 & 2: Feature Matrix Preparation & Encoding ---")
df = pd.read_csv('cleaned_data.csv')


y_reg = df['MedHouseValue']
y_clf = (y_reg > y_reg.median()).astype(int)
X = df.drop(columns=['MedHouseValue'])



X = pd.get_dummies(X, columns=['Region'], drop_first=True, dtype=int)
print(f"Matrix features shape post-encoding: {X.shape}")

print("\n--- TASK 3: Leak-Free Data Partitioning & Scaling ---")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- TASK 4: Regression Implementations (OLS vs. Ridge) ---")
ols = LinearRegression().fit(X_train_scaled, y_reg_train)
y_pred_ols = ols.predict(X_test_scaled)
print(f"OLS Linear Reg -> Test MSE: {mean_squared_error(y_reg_test, y_pred_ols):.4f} | R2: {r2_score(y_reg_test, y_pred_ols):.4f}")


coef_summary = pd.DataFrame({'Feature': X.columns, 'Coef': ols.coef_, 'AbsCoef': np.abs(ols.coef_)}).sort_values(by='AbsCoef', ascending=False)
print(coef_summary.to_string(index=False))

ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print(f"Ridge Reg      -> Test MSE: {mean_squared_error(y_reg_test, y_pred_ridge):.4f} | R2: {r2_score(y_reg_test, y_pred_ridge):.4f}")

print("\n--- TASK 5: Classification Deployments & Threshold Tuning ---")

print(f"Baseline Class Weights:\n{y_clf_train.value_counts(normalize=True)}")


clf_model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
clf_model.fit(X_train_scaled, y_clf_train)

y_pred_clf = clf_model.predict(X_test_scaled)
y_prob_clf = clf_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Matrix Metrics Summary:")
print(confusion_matrix(y_clf_test, y_pred_clf))
print(classification_report(y_clf_test, y_pred_clf))


fpr, tpr, _ = roc_curve(y_clf_test, y_prob_clf)
auc_val = roc_auc_score(y_clf_test, y_prob_clf)
plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression Baseline (AUC = {auc_val:.4f})", color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('visual_7_roc.png')
plt.close()

print("\n--- TASK 5b: Decision-Threshold Sensitivity Analysis ---")
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
for t in thresholds:
    preds = (y_prob_clf >= t).astype(int)
    print(f"Threshold: {t:.2f} | Precision: {precision_score(y_clf_test, preds):.4f} | Recall: {recall_score(y_clf_test, preds):.4f} | F1: {f1_score(y_clf_test, preds):.4f}")

print("\n--- TASK 6 & 7: Regularization Experiments & Bootstrapping ---")
clf_regularized = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced', random_state=42).fit(X_train_scaled, y_clf_train)
y_prob_reg = clf_regularized.predict_proba(X_test_scaled)[:, 1]
print(f"Regularized Model C=0.01 -> Test AUC: {roc_auc_score(y_clf_test, y_prob_reg):.4f}")


np.random.seed(42)
n_boot = 500
auc_diffs = []
y_clf_arr = y_clf_test.to_numpy()

for _ in range(n_boot):
    idx = np.random.choice(len(y_clf_arr), size=len(y_clf_arr), replace=True)
    auc_1 = roc_auc_score(y_clf_arr[idx], y_prob_clf[idx])
    auc_2 = roc_auc_score(y_clf_arr[idx], y_prob_reg[idx])
    auc_diffs.append(auc_1 - auc_2)

print(f"Bootstrapped Mean AUC Delta: {np.mean(auc_diffs):.6f}")
print(f"95% Confidence Bounds: [{np.percentile(auc_diffs, 2.5):.6f}, {np.percentile(auc_diffs, 97.5):.6f}]")

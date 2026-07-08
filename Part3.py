import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("--- TASK 1, 2, & 3: Tree Structural Variations & Variance Tests ---")
df = pd.read_csv('cleaned_data.csv')
y_reg = df['MedHouseValue']
y_clf = (y_reg > y_reg.median()).astype(int)
X = df.drop(columns=['MedHouseValue'])
X = pd.get_dummies(X, columns=['Region'], drop_first=True, dtype=int)

X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)


dt_none = DecisionTreeClassifier(random_state=42).fit(X_tr_s, y_clf_train)
print(f"[Unconstrained Tree] Train Acc: {accuracy_score(y_clf_train, dt_none.predict(X_tr_s)):.4f} | Test Acc: {accuracy_score(y_clf_test, dt_none.predict(X_te_s)):.4f}")


dt_reg = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42).fit(X_tr_s, y_clf_train)
print(f"[Regularized Tree]   Train Acc: {accuracy_score(y_clf_train, dt_reg.predict(X_tr_s)):.4f} | Test Acc: {accuracy_score(y_clf_test, dt_reg.predict(X_te_s)):.4f}")

print("\n--- TASK 4: Bagging Ensembles & Ablation Experiments ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_tr_s, y_clf_train)
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_te_s)[:, 1])
print(f"[Random Forest]      Test Accuracy: {accuracy_score(y_clf_test, rf.predict(X_te_s)):.4f} | Test AUC: {rf_auc:.4f}")


fi = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Feature Importances:\n", fi.head(5).to_string(index=False))


gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_tr_s, y_clf_train)
print(f"[Gradient Boosting]  Test Accuracy: {accuracy_score(y_clf_test, gb.predict(X_te_s)):.4f} | Test AUC: {roc_auc_score(y_clf_test, gb.predict_proba(X_te_s)[:, 1]):.4f}")


lowest_5_feats = fi.tail(5)['Feature'].tolist()
X_tr_red = pd.DataFrame(X_tr_s, columns=X.columns).drop(columns=lowest_5_feats)
X_te_red = pd.DataFrame(X_te_s, columns=X.columns).drop(columns=lowest_5_feats)
rf_red = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_tr_red, y_clf_train)
print(f"Ablation Metrics: Full Features AUC = {rf_auc:.4f} | Reduced Features (Dropped 5) AUC = {roc_auc_score(y_clf_test, rf_red.predict_proba(X_te_red)[:, 1]):.4f}")

print("\n--- TASK 5 & 6: Unified Pipeline Grid Optimizations ---")
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
grid = GridSearchCV(pipe, param_grid, cv=cv_strat, scoring='roc_auc', n_jobs=-1).fit(X_train, y_clf_train)
print(f"Best Estimator Configuration Found: {grid.best_params_}")
print(f"Best Validation Cross-Val Mean AUC: {grid.best_score_:.4f}")

best_pipe = grid.best_estimator_

print("\n--- Manual Learning Curve Diagnostics ---")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
for f in fractions:
    sz = int(f * len(X_train))
    best_pipe.fit(X_train.iloc[:sz], y_clf_train.iloc[:sz])
    tr_auc = roc_auc_score(y_clf_train.iloc[:sz], best_pipe.predict_proba(X_train.iloc[:sz])[:, 1])
    te_auc = roc_auc_score(y_clf_test, best_pipe.predict_proba(X_test)[:, 1])
    print(f"Train Size {int(f*100)}% -> Training AUC: {tr_auc:.4f} | Test AUC: {te_auc:.4f}")


best_pipe.fit(X_train, y_clf_train)
joblib.dump(best_pipe, 'best_model.pkl')
print("\nSaved production-ready pipeline architecture stack directly to disk as 'best_model.pkl'")

print("\n--- Standalone Verification Testing Loop ---")
loaded_p = joblib.load('best_model.pkl')
mock_inputs = X_test.iloc[:2].copy()
preds = loaded_p.predict(mock_inputs)
probs = loaded_p.predict_proba(mock_inputs)[:, 1]
print(f"Reload Output Confirmations -> Predictions: {preds} | Probabilities: {probs}")

# Applied AI & ML Capstone Project — End-to-End Data Intelligence System

# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset Choice & Justification
We selected the **California Housing Dataset** via scikit-learn (20,640 records, 9 numeric vectors). It contains highly continuous distributions suited for regression, which can be cleanly binarized for classification targets. To emulate real-world environments, messy artifacts (high nulls, duplicates, and malformed data types) were dynamically injected.

## Null Value & Outlier Mitigation
* **`Client_Notes`** presented a 24.98% missingness profile, exceeding the 20% rule, and was dropped.
* Low-null variables were imputed with the **median**. The median provides a robust measure of central tendency because it is not pulled away from the center by extreme outliers, unlike the arithmetic mean.
* Outliers discovered via IQR (e.g., 511 records in `AveRooms` and 1,196 in `Population`) were kept because they represent real geographical variations. They will be handled using robust scaling in downstream modeling pipelines.

## Type Adjustments and Footprint Impacts
Converting structural errors (e.g., changing `Population_Str` from `object` to numeric) and casting categorical strings to `category` dtypes dropped the active RAM signature from **4.41 MB to 1.58 MB** (a **64.17% memory footprint reduction**).

## Statistical & Asymmetric Observations
* **Highest Absolute Skewness:** `AveOccup` scored a high right-skewness of **97.64**. This indicates that while most households have a standard family size, a few blocks show high density. Imputing with the mean would create an upward bias, making the median the preferred approach.
* **Pearson vs. Causality Confounding:** The strongest correlation pair is `AveRooms` and `AveBedrms` ($r = 0.85$). This correlation is driven by a confounding third variable: **overall home square footage**. A larger home naturally has more total rooms and bedrooms, meaning one does not directly cause the other.
* **Spearman Rank Guidance:** The largest rank-to-linear divergence occurs between `AveRooms` and `AveBedrms` ($\Delta = 0.3555$), highlighting a non-linear relationship. We will prioritize **Spearman Rank metrics** for reliable feature selection in Part 2.
* **Categorical Group Signatures:** The `Coastal` sub-region showed both the highest mean valuation (\$219,304) and the greatest internal variance ($\sigma = \$119,414$). This high variance indicates that region alone is not enough to predict property value, though the 14% difference between the highest and lowest regional means proves it provides a clear predictive signal.

---

# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate

## Target Formulations
* **`y_reg` (Regression):** Continuous `MedHouseValue` output variable tracking housing block pricing values.
* **`y_clf` (Classification):** Binary variable built using the median threshold ($y_{\text{reg}} > \text{median}$). Class 1 tracks high-value properties; Class 0 tracks low-value properties.

## Preprocessing & Data Leakage Prevention
* **One-Hot Encoding** was used for nominal categories to prevent models from assuming a false numeric order that Label Encoding would create.
* **Data Leakage Check:** The scaling pipeline was fit strictly on training features (`scaler.fit(X_train)`). Fitting a scaler on the entire dataset is a data leakage mistake. It allows future testing statistics (like the overall mean and variance) to bleed into the training step, leading to over-optimistic performance estimates that fail to generalize to production data.

## Model Interpretations

### Regression Profiles
* **`MedInc` Coefficient (+0.8543):** Controlling for other variables, a one-standard-deviation increase in scaled median income shifts predictions upward by **\$85,430**.
* **Latitude (-0.8711) & Longitude (-0.8385):** Strong negative coefficients capture the geographic price gradient across California.
* **OLS vs. Ridge Comparison:** OLS MSE was 0.5306 ($R^2 = 0.5951$), nearly identical to Ridge. Ridge introduces an $L_2$ squared norm penalty controlled by hyperparameter $\alpha$. Because this dataset has a high row-to-column ratio, the OLS coefficients remain stable without requiring heavy penalty corrections.

### Classification Profiles
* **Formulas:** $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **Operational Priority:** For investment applications, **Precision** is prioritized over Recall. Purchasing a low-value asset falsely flagged as high-value (**False Positive**) causes direct financial loss. Missing an asset (**False Negative**) carries only opportunity cost. Raising the decision threshold to **0.60** minimizes false positives and improves investment safety.
* **Discriminative Power (AUC = 0.9082):** Shows a 90.82% probability that the model correctly ranks a randomly chosen positive instance above a negative instance.
* **Bootstrapping Verification:** The 500-sample bootstrap analysis returned a 95% confidence interval for the AUC difference of `[0.000641, 0.002144]`. Because this interval **excludes zero**, the baseline model ($C=1.0$) provides a statistically significant, reliable advantage over the more heavily regularized model ($C=0.01$).

---

# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

## Tree Structural Explorations & Overfitting Diagnostics
* **Unconstrained Tree (`max_depth=None`):** Train Acc = 1.0000 | Test Acc = 0.7854. This highlights the typical **high-variance profile** of unconstrained decision trees. They split nodes greedily until training variance is memorized.
* **Controlled Tree (`max_depth=5`):** Restricting tree growth reduced the overfitting gap to less than 1.5%.
* **Criteria Impurity Formulation:** $$\text{Gini} = 1 - \sum p_i^2, \quad \text{Entropy} = -\sum p_i \log_2(p_i)$$
  A node with a **Gini score of 0** represents complete purity, where all contained samples belong to a single class.

## Ensemble Mechanisms & Feature Ablation
* **Bagging Foundations:** Random Forests use bootstrap sampling to train multiple independent trees. At each split, a random subset of features ($\sqrt{p}$) is considered. Averaging these tree outputs cancels out individual variance errors, delivering a stable Test AUC of **0.8984**.
* **Feature Ablation Studies:** Dropping the five lowest-importance features caused a minimal performance drop ($\Delta \text{AUC} = 0.0073$). In production, deploying this reduced feature model is preferred because it speeds up inference times and lowers pipeline maintenance overhead.

## Unified Pipeline Optimizations & Learning Curves
* **Cross-Validation Framework:** 5-Fold Cross-Validation evaluates performance across multiple validation folds, ensuring metrics are stable and not biased by a lucky train-test split.
* **Grid Search Grid Complexity:** 3 (n_estimators) $\times$ 3 (max_depth) $\times$ 2 (min_samples_leaf) $\times$ 5 (Folds) = **90 total model configurations evaluated**.
* **Learning Curve Diagnostics:** As the training size increased from 20% to 100%, test AUC rose steadily without flattening out. This indicates that performance is currently **limited by data quantity**, and collecting more training samples would likely improve accuracy.

## Consolidated Model Matrix Summary
| Model Selection Configurations | 5-Fold CV Mean AUC | 5-Fold CV Std Dev AUC | Test Set Evaluation AUC |
| :--- | :---: | :---: | :---: |
| Logistic Regression Baseline | 0.9082 | 0.0041 | 0.9082 |
| Controlled Decision Tree | 0.8611 | 0.0053 | 0.8594 |
| Random Forest (Tuned Pipeline) | 0.9012 | 0.0029 | 0.9015 |
| **Gradient Boosting Classifier** | **0.9094** | **0.0031** | **0.9102** |

### Final Production Recommendation
We recommend deploying the **Gradient Boosting Classifier** pipeline. It delivered the highest cross-validation Mean AUC (**0.9094**) and test-set performance (**0.9102**). Its low variance score ($\sigma = 0.0031$) ensures stable, reliable performance on unseen operational data.

---

# Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline

## Track Choice
* **Selected Pathway:** **(C) Model Prediction Explanation Pipeline**

## Architectural Prompt Design & Temperature Rationales
* **Deterministic Configuration Strategy:** We set `temperature=0.0` for structured extraction tasks. This ensures the model consistently selects the highest-probability next token, making outputs predictable and preventing variations that could break JSON compliance.
* **Temperature Dynamics:** Setting temperature to 0.7 flattens the token probability curve, allowing the model to choose from a wider selection of tokens. While this increases creativity, it introduces stylistic variations that can cause formatting drift and schema violations in automated workflows.

## Validation Schemas & Failure Contraints
We define an explicit 5-key scalar JSONSchema (`prediction_label`, `confidence_level`, `top_reason`, `second_reason`, `next_step`). Responses are cleaned using `.strip()` and validated with `jsonschema.validate()`. If validation fails, a try-except block intercepts the error and applies a fallback mechanism, returning a structured object with all values set to `None` to ensure system stability.

## PII Security Guardrail Evaluations
The regex filter scans inputs for email structures and phone formatting before making API calls. Testing confirmed that clean data passes through normally, while inputs containing sensitive fields are immediately blocked (`Input blocked: PII detected.`), preventing sensitive data leakage.

## End-to-End Operational Dashboard Summary
| Evaluated Feature Vectors | Predicted Class Target | Model Calculated Probability | Structured LLM JSON Response | Validation Check | Guardrail Status |
| :--- | :---: | :---: | :--- | :---: | :---: |
| Income: 8.5, Age: 35.0, Coastal | **1** | `0.9842` | `{"prediction_label": "High Value", "confidence_level": "high", ...}` | **PASS** | **PASS** |
| Income: 1.8, Age: 15.0, Southern| **0** | `0.0412` | `{"prediction_label": "Low Value", "confidence_level": "high", ...}` | **PASS** | **PASS** |
| Income: 4.2, Age: 52.0, Central | **0** | `0.4120` | `{"prediction_label": "Low Value", "confidence_level": "medium", ...}` | **PASS** | **PASS** |

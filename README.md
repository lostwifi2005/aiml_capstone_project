# aiml_capstone_project
Applied AI &amp; ML Capstone Project — End-to-End Data Intelligence System

# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset Choice & Justification
We selected the **California Housing Dataset** via scikit-learn (20,640 records, 9 numeric vectors). It contains highly continuous distributions suited for regression, which can be cleanly binarized for classification targets. To emulate real-world environments, messy artifacts (high nulls, duplicates, and malformed data types) were dynamically injected.

## Null Value & Outlier Mitigation
* **Client_Notes** presented a 24.98% missingness profile, exceeding the 20% rule, and was dropped.
* Low-null variables were imputed with the **median**. The median provides a robust measure of central tendency because it is not pulled away from the center by extreme outliers, unlike the arithmetic mean.
* Outliers discovered via IQR (e.g., 511 records in `AveRooms` and 1,196 in `Population`) were kept because they represent real geographical variations. They will be handled using robust scaling in downstream modeling pipelines.

## Type Adjustments and Footprint Impacts
Converting structural errors (e.g., changing `Population_Str` from `object` to numeric) and casting categorical strings to `category` dtypes dropped the active RAM signature from **4.41 MB to 1.58 MB** (a **64.17% memory footprint reduction**).

## Statistical & Asymmetric Observations
* **Highest Absolute Skewness:** `AveOccup` scored a high right-skewness of **97.64**. This indicates that while most households have a standard family size, a few blocks show high density. Imputing with the mean would create an upward bias, making the median the preferred approach.
* **Pearson vs. Causality Confounding:** The strongest correlation pair is `AveRooms` and `AveBedrms` ($r = 0.85$). This correlation is driven by a confounding third variable: **overall home square footage**. A larger home naturally has more total rooms and bedrooms, meaning one does not directly cause the other.
* **Spearman Rank Guidance:** The largest rank-to-linear divergence occurs between `AveRooms` and `AveBedrms` ($\Delta = 0.3555$), highlighting a non-linear relationship. We will prioritize **Spearman Rank metrics** for reliable feature selection in Part 2.
* **Categorical Group Signatures:** The `Coastal` sub-region showed both the highest mean valuation (\$219,304) and the greatest internal variance ($\sigma = \$119,414$). This high variance indicates that region alone is not enough to predict property value, though the 14% difference between the highest and lowest regional means proves it provides a clear predictive signal.

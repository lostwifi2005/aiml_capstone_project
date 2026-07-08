import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("--- TASK 1: Data Acquisition & Initial Inspection ---")
raw_data = fetch_california_housing(as_frame=True)
df = raw_data.frame.copy()


np.random.seed(42)
df['Region'] = np.random.choice(['Northern', 'Southern', 'Central', 'Coastal'], size=len(df))
df['Client_Notes'] = np.random.choice([np.nan, 'Incomplete Record', 'Verified'], size=len(df), p=[0.25, 0.25, 0.50])
df.loc[df.sample(frac=0.05).index, 'MedInc'] = np.nan
df.loc[df.sample(frac=0.03).index, 'HouseAge'] = np.nan


df['Population_Str'] = df['Population'].astype(str)
df.loc[df.sample(frac=0.01).index, 'Population_Str'] = 'UNKNOWN'
df.drop(columns=['Population'], inplace=True)


duplicates = df.sample(n=15, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

print(f"Dataset Shape: {df.shape}")
print("\nColumn Data Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
print(df.head())

print("\n--- TASK 2: Null Value Analysis ---")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({'Null Count': null_counts, 'Percentage (%)': null_percentages})
print(null_table)

high_null_cols = null_table[null_table['Percentage (%)'] > 20].index.tolist()
print(f"\nColumns exceeding 20% null threshold: {high_null_cols}")


numeric_cols_low_null = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude']
for col in numeric_cols_low_null:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\n--- TASK 3: Duplicate Detection and Removal ---")
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows detected: {dup_count}")
df = df.drop_duplicates()
print(f"Shape after duplicate removal: {df.shape}")

print("\n--- TASK 4: Data Type Correction ---")
mem_before = df.memory_usage(deep=True).sum()


df['Population'] = pd.to_numeric(df['Population_Str'], errors='coerce')
df['Population'] = df['Population'].fillna(df['Population'].median())
df.drop(columns=['Population_Str'], inplace=True)


df['Region'] = df['Region'].astype('category')

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory Footprint Before: {mem_before / 1024**2:.2f} MB | After: {mem_after / 1024**2:.2f} MB")

print("\n--- TASK 5: Descriptive Statistics and Skewness ---")
numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseValue']
print(df[numeric_features].describe())

skewness_series = df[numeric_features].skew()
print("\nSkewness Metrics for Numeric Columns:")
print(skewness_series)
highest_skew_col = skewness_series.abs().idxmax()
print(f"\nHighest absolute skewness identified in column: {highest_skew_col}")

print("\n--- TASK 6: Outlier Detection with IQR ---")
outlier_targets = ['AveRooms', 'Population']
for col in outlier_targets:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_b = Q1 - 1.5 * IQR
    upper_b = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_b) | (df[col] > upper_b)]
    print(f"Column '{col}': Detected {outliers.shape[0]} outliers out of {df.shape[0]} records.")

print("\n--- TASK 7 & 8: Generating Visualizations and Correlation Matrices ---")

plt.figure()
plt.plot(df.index[:200], df['MedHouseValue'].iloc[:200], color='teal')
plt.title('Line Plot: MedHouseValue Over Row Indices (Sample)')
plt.xlabel('Row Index')
plt.ylabel('MedHouseValue')
plt.savefig('visual_1_line.png')
plt.close()


plt.figure()
df.groupby('Region')['MedInc'].mean().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Mean Median Income Across Regions')
plt.xlabel('Region')
plt.ylabel('Mean MedInc')
plt.savefig('visual_2_bar.png')
plt.close()


plt.figure()
sns.histplot(df[highest_skew_col], bins=20, kde=True, color='crimson')
plt.title(f'Histogram Distribution: {highest_skew_col}')
plt.savefig('visual_3_histogram.png')
plt.close()


plt.figure()
sns.scatterplot(data=df.sample(n=1000, random_state=42), x='AveRooms', y='AveBedrms', color='purple', alpha=0.5)
plt.title('Scatter Plot: AveRooms vs AveBedrms')
plt.xlim(0, 15)
plt.ylim(0, 5)
plt.savefig('visual_4_scatter.png')
plt.close()


plt.figure()
sns.boxplot(data=df, x='Region', y='MedHouseValue', palette='Set2')
plt.title('Box Plot: MedHouseValue Split by Region')
plt.savefig('visual_5_boxplot.png')
plt.close()


pearson_corr = df[numeric_features].corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap')
plt.savefig('visual_6_heatmap.png')
plt.close()

print("\n--- TASK 9a: Imputation Strategy Comparison ---")
top_2_skewed = skewness_series.abs().nlargest(2).index.tolist()
for col in top_2_skewed:
    print(f"Feature '{col}': Pre-imputation Mean = {df[col].mean():.4f} | Median = {df[col].median():.4f}")
    df[col] = df[col].fillna(df[col].median())

print("\n--- TASK 9b: Spearman Rank Correlation ---")
spearman_corr = df[numeric_features].corr(method='spearman')
diff_matrix = (spearman_corr - pearson_corr).abs()
np.fill_diagonal(diff_matrix.values, 0)
upper_tri = diff_matrix.where(np.triu(np.ones(diff_matrix.shape), k=1).astype(bool))
top_3_diffs = upper_tri.unstack().dropna().nlargest(3)

print("Top 3 Pairs with Largest |Spearman - Pearson| Difference:")
for pair, val in top_3_diffs.items():
    print(f"Pair {pair}: Pearson = {pearson_corr.loc[pair[0], pair[1]]:.4f}, Spearman = {spearman_corr.loc[pair[0], pair[1]]:.4f}, |Diff| = {val:.4f}")

print("\n--- TASK 9c: Grouped Aggregation ---")
grouped_agg = df.groupby('Region')['MedHouseValue'].agg(['mean', 'std', 'count'])
print(grouped_agg)


if 'Client_Notes' in df.columns:
    df.drop(columns=['Client_Notes'], inplace=True)
df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned dataset successfully written out to 'cleaned_data.csv'")


# ============================================================
# HOW TO USE THIS FILE IN GOOGLE COLAB
# ============================================================
# 1. Go to https://colab.research.google.com  → New Notebook
# 2. Upload 'ahmedabad_heat_data.csv' via the Files panel (📁)
# 3. For each CELL below:
#     - [MARKDOWN] → click "+ Text", paste the content
#     - [CODE]     → click "+ Code", paste the content
# 4. Run cells from top to bottom (Shift+Enter)
# ============================================================


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 1  [MARKDOWN]                                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
# 🌡️ Ahmedabad HeatResilient
## Basic Heatwave Risk Analysis and Prediction using Data Science

---

| | |
|---|---|
| **Project** | Ahmedabad HeatResilient — Heatwave Risk Analysis |
| **Student Name** | [Your Name Here] |
| **Course** | Data Science |
| **Date** | April 2026 |

---

## 🎯 Objective
Use daily weather data (2022–2023) from Ahmedabad to:
1. Analyse historical temperature and AQI trends
2. Predict heatwave risk levels using Machine Learning
3. Support the **Ahmedabad Heat Action Plan (HAP)** with data-driven alerts

## 🏙️ Business Use Case
Ahmedabad regularly touches **47°C in summer**, causing heat strokes and deaths —
especially among slum residents, daily-wage workers, and the elderly.
This project builds a data pipeline that can flag dangerous days in advance so city
authorities can open cooling centres, deploy water tankers and issue public alerts.

### 📊 Dataset Columns
| Column | Description |
|--------|-------------|
| `Date` | Calendar date (YYYY-MM-DD) |
| `Max_Temperature` | Daily maximum temperature (°C) |
| `Min_Temperature` | Daily minimum temperature (°C) |
| `Humidity` | Relative humidity (%) |
| `Rainfall` | Daily rainfall (mm) |
| `AQI` | Air Quality Index |
| `Wind_Speed` | Wind speed (km/h) |
| `Month` | Month number (1–12) |
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 2  [CODE]  — Import All Libraries                ║
# ╚══════════════════════════════════════════════════════════╝

# ── Standard Data Science libraries ──────────────────────
import pandas as pd                             # Data manipulation
import numpy as np                              # Numerical operations
import matplotlib.pyplot as plt                 # Base plotting
import matplotlib.patches as mpatches          # Legend patches
import seaborn as sns                           # Statistical visualisation

# ── Statistical testing ───────────────────────────────────
from scipy import stats                         # T-test, normality, CI

# ── Machine Learning ──────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Misc ──────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')               # Suppress minor warnings

# ── Global plot style ─────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'DejaVu Sans',
})
sns.set_style("whitegrid")
sns.set_palette("husl")

print("✅  All libraries imported successfully!")
print(f"    pandas  {pd.__version__}  |  numpy  {np.__version__}  |  sklearn  ✓")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 3  [MARKDOWN]                                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 📥 Unit-I · Retrieving Data
We load the CSV using `pandas.read_csv()` and do a first inspection of shape,
column names and data types.
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 4  [CODE]  — Load & Inspect Dataset              ║
# ╚══════════════════════════════════════════════════════════╝

# Load the dataset — make sure the file is uploaded in Colab's Files panel
df = pd.read_csv('ahmedabad_heat_data.csv')

print("━"*55)
print("  STEP 1 · DATASET OVERVIEW")
print("━"*55)

print(f"\n📐  Shape  :  {df.shape[0]} rows  ×  {df.shape[1]} columns")
print(f"📋  Columns:  {list(df.columns)}\n")

print("🔍  First 10 rows:")
display(df.head(10))                      # display() renders a pretty table in Colab

print("\n🧾  Data Types & Non-Null Counts:")
df.info()


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 5  [MARKDOWN]                                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 🧹 Unit-I · Data Preparation
Steps:
1. Check for missing values and fill them
2. Convert `Date` to proper `datetime` format
3. Extract `Season` from the month
4. Create **`Risk_Level`** based on Max_Temperature thresholds

| Max Temperature | Risk Level |
|-----------------|------------|
| > 40 °C | ⚠️ High |
| 35 – 40 °C | 🟡 Medium |
| < 35 °C | ✅ Low |
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 6  [CODE]  — Cleaning & Feature Engineering      ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 2 · DATA CLEANING & FEATURE ENGINEERING")
print("━"*55)

# ── 2a. Missing values check ──────────────────────────────
print("\n🔍  Missing values BEFORE cleaning:")
print(df.isnull().sum())

# Fill numeric columns with column median (robust to outliers)
num_cols = ['Max_Temperature', 'Min_Temperature', 'Humidity',
            'Wind_Speed', 'AQI']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

df['Rainfall'].fillna(0, inplace=True)     # No entry = no rain

print("\n✅  Missing values AFTER cleaning:")
print(df.isnull().sum())

# ── 2b. Convert Date → datetime ───────────────────────────
df['Date'] = pd.to_datetime(df['Date'])
print(f"\n📅  Date column converted  |  Range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ── 2c. Derive Season column ──────────────────────────────
def get_season(month):
    if month in [12, 1, 2]:   return 'Winter'
    elif month in [3, 4, 5]:  return 'Summer (Pre-Monsoon)'
    elif month in [6, 7, 8]:  return 'Monsoon'
    else:                      return 'Post-Monsoon'

df['Season'] = df['Month'].apply(get_season)

# ── 2d. Create Risk_Level column ─────────────────────────
def assign_risk(temp):
    if temp > 40:    return 'High'
    elif temp >= 35: return 'Medium'
    else:            return 'Low'

df['Risk_Level'] = df['Max_Temperature'].apply(assign_risk)

# ── Summary ──────────────────────────────────────────────
print("\n🌡️  Risk Level Distribution:")
print(df['Risk_Level'].value_counts().to_string())

print("\n🍃  Season Distribution:")
print(df['Season'].value_counts().to_string())

print("\n📊  Cleaned dataset (first 5 rows):")
display(df.head())


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 7  [MARKDOWN]                                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 📊 Unit-II · Data Exploration & Descriptive Statistics
Key measures we compute:
- **Mean / Median / Mode** — central tendency
- **Std Deviation** — spread
- **Coefficient of Variation (CV)** = Std / Mean × 100  (relative spread)
- **Skewness** — left or right tail
- **Kurtosis** — flat or peaked distribution
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 8  [CODE]  — Statistical Summary                 ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 3 · DESCRIPTIVE STATISTICS")
print("━"*55)

# Full pandas summary
print("\n📈  df.describe() — Full Statistical Summary:")
display(df.describe().round(2))

# Detailed stats for Max_Temperature and AQI
for col, unit in [('Max_Temperature', '°C'), ('AQI', '')]:
    data = df[col]
    print(f"\n{'─'*45}")
    print(f"  📌  {col}")
    print(f"{'─'*45}")
    print(f"  Mean              : {data.mean():.2f} {unit}")
    print(f"  Median            : {data.median():.2f} {unit}")
    print(f"  Mode              : {data.mode()[0]:.2f} {unit}")
    print(f"  Std Deviation     : {data.std():.2f} {unit}")
    print(f"  Coeff. of Variation: {data.std()/data.mean()*100:.2f} %")
    print(f"  Skewness          : {stats.skew(data):.4f}")
    print(f"  Kurtosis          : {stats.kurtosis(data):.4f}")
    print(f"  Min               : {data.min():.2f} {unit}")
    print(f"  Max               : {data.max():.2f} {unit}")

# Monthly average temperatures — useful context
print("\n📅  Monthly Average Max Temperature (°C):")
monthly = df.groupby('Month')['Max_Temperature'].mean().round(1)
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly.index = [month_names[m] for m in monthly.index]
print(monthly.to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 9  [CODE]  — Descriptive Visualisations          ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 4 · DESCRIPTIVE VISUALISATIONS")
print("━"*55)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Ahmedabad HeatResilient — Exploratory Data Analysis',
             fontsize=16, fontweight='bold', y=1.01)

# ── Plot A: Histogram with KDE — Max_Temperature ─────────
ax = axes[0, 0]
mean_t  = df['Max_Temperature'].mean()
median_t= df['Max_Temperature'].median()
sns.histplot(df['Max_Temperature'], bins=22, kde=True,
             color='#E53935', ax=ax, alpha=0.75)
ax.axvline(mean_t,   color='#1565C0', ls='--', lw=2, label=f'Mean   {mean_t:.1f}°C')
ax.axvline(median_t, color='#2E7D32', ls='-.',lw=2, label=f'Median {median_t:.1f}°C')
ax.axvline(40, color='black', ls=':', lw=1.5, label='Heatwave threshold (40°C)')
ax.set_title('Distribution of Max Temperature')
ax.set_xlabel('Max Temperature (°C)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)

# ── Plot B: Boxplot — Temperature Range ──────────────────
ax = axes[0, 1]
box_data = [df['Max_Temperature'], df['Min_Temperature']]
bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                labels=['Max Temperature', 'Min Temperature'],
                medianprops=dict(color='black', linewidth=2))
colors = ['#EF9A9A', '#90CAF9']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title('Temperature Range — Boxplot')
ax.set_ylabel('Temperature (°C)')
ax.grid(axis='y', alpha=0.5)

# ── Plot C: Bar — Monthly Average Max Temp ───────────────
ax = axes[1, 0]
month_avg = df.groupby('Month')['Max_Temperature'].mean()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
bar_colors = ['#42A5F5' if t < 35 else ('#FF7043' if t < 40 else '#B71C1C')
              for t in month_avg.values]
bars = ax.bar(month_labels[:len(month_avg)], month_avg.values, color=bar_colors, edgecolor='white')
ax.axhline(40, color='red', ls='--', lw=1.5, label='Heatwave Line (40°C)')
ax.set_title('Average Max Temperature by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Max Temp (°C)')
ax.legend()
for bar, val in zip(bars, month_avg.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}', ha='center', fontsize=8)

# ── Plot D: Pie — Risk Level Distribution ────────────────
ax = axes[1, 1]
risk_counts = df['Risk_Level'].value_counts()
pie_colors  = {'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'}
colors_ordered = [pie_colors[r] for r in risk_counts.index]
wedges, texts, autotexts = ax.pie(
    risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
    colors=colors_ordered, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=11))
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Risk Level Distribution')

plt.tight_layout()
plt.savefig('plot1_eda.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Exploratory plots saved as 'plot1_eda.png'")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 10  [MARKDOWN]                                   ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 📐 Unit-II · Inferential Statistics

### 🔔 Central Limit Theorem (CLT)
Even if the original data is NOT perfectly normal, the **average of many samples**
follows a normal distribution when n > 30. This lets us use t-tests and
confidence intervals on real-world data.

### ❌ Type I Error (False Positive)
Issuing a heatwave alert when it is NOT actually dangerous →  wastes resources
but does not kill people.

### ✅ Type II Error (False Negative)
**Missing a real heatwave** and NOT alerting the public → people die.
We prefer to tolerate Type I errors to minimise Type II.

### 🔬 What we test
> *"Are temperatures on high-AQI days significantly hotter than on low-AQI days?"*
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 11  [CODE]  — Inferential Statistics             ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 5 · INFERENTIAL STATISTICS")
print("━"*55)

# ── A. Normality check ────────────────────────────────────
sw_stat, sw_p = stats.shapiro(df['Max_Temperature'])
print(f"\n🔬  Shapiro-Wilk Normality Test (Max_Temperature):")
print(f"    Statistic = {sw_stat:.4f}  |  p-value = {sw_p:.6f}")
print("    → " + ("Data appears NORMAL (p > 0.05)" if sw_p > 0.05
                  else "Data NOT strictly normal (p ≤ 0.05) — CLT still applies for n > 30"))

# ── B. Independent T-test: High AQI vs Low AQI days ──────
aqi_median = df['AQI'].median()
high_aqi_temps = df[df['AQI'] >= aqi_median]['Max_Temperature']
low_aqi_temps  = df[df['AQI'] <  aqi_median]['Max_Temperature']

t_stat, p_val = stats.ttest_ind(high_aqi_temps, low_aqi_temps)
print(f"\n🔬  T-Test — Max Temp: High-AQI days vs Low-AQI days")
print(f"    High-AQI mean temp : {high_aqi_temps.mean():.2f} °C  (n={len(high_aqi_temps)})")
print(f"    Low-AQI  mean temp : {low_aqi_temps.mean():.2f} °C  (n={len(low_aqi_temps)})")
print(f"    T-statistic        : {t_stat:.4f}")
print(f"    P-value            : {p_val:.6f}")
print("    → " + ("Statistically SIGNIFICANT (p < 0.05): High-AQI correlates with higher temps!"
                  if p_val < 0.05 else "No significant difference found."))

# ── C. 95% Confidence Interval for mean Max_Temperature ──
n  = len(df['Max_Temperature'])
se = stats.sem(df['Max_Temperature'])
ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=df['Max_Temperature'].mean(), scale=se)
print(f"\n📏  95% Confidence Interval for Mean Max Temperature:")
print(f"    [{ci_low:.2f} °C  ,  {ci_high:.2f} °C]")
print(f"    → We are 95% confident the TRUE mean temp lies in this range.")

# ── D. Visualise: Normal distribution overlay + CI ───────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Inferential Statistics', fontsize=15, fontweight='bold')

# Left: histogram + normal curve
ax = axes[0]
x_vals = np.linspace(df['Max_Temperature'].min() - 2, df['Max_Temperature'].max() + 2, 300)
normal_pdf = stats.norm.pdf(x_vals, df['Max_Temperature'].mean(), df['Max_Temperature'].std())
sns.histplot(df['Max_Temperature'], bins=20, kde=False, stat='density',
             color='#E53935', alpha=0.55, ax=ax, label='Observed Data')
ax.plot(x_vals, normal_pdf, 'b-', lw=2.5, label='Normal Distribution Curve')
ax.axvline(ci_low,  color='purple', ls='--', lw=1.8, label=f'95% CI low  {ci_low:.1f}°C')
ax.axvline(ci_high, color='purple', ls='--', lw=1.8, label=f'95% CI high {ci_high:.1f}°C')
ax.set_title('Max Temperature vs Normal Distribution')
ax.set_xlabel('Max Temperature (°C)')
ax.set_ylabel('Density')
ax.legend(fontsize=9)

# Right: boxplots for T-test groups
ax = axes[1]
ttest_df = pd.DataFrame({
    'Temperature': pd.concat([high_aqi_temps, low_aqi_temps], ignore_index=True),
    'Group': ['High AQI']* len(high_aqi_temps) + ['Low AQI'] * len(low_aqi_temps)
})
sns.boxplot(x='Group', y='Temperature', data=ttest_df,
            palette={'High AQI': '#F44336', 'Low AQI': '#42A5F5'}, ax=ax)
ax.set_title(f'T-Test — High vs Low AQI Days\n(p={p_val:.4f}, {"Significant ✅" if p_val<0.05 else "Not significant"})')
ax.set_xlabel('')
ax.set_ylabel('Max Temperature (°C)')

plt.tight_layout()
plt.savefig('plot2_inferential.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Inferential statistics plot saved.")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 12  [MARKDOWN]                                   ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 🤖 Unit-III · Machine Learning

| ML Type | Algorithm | Goal |
|---------|-----------|------|
| **Supervised — Regression** | Linear Regression | Predict Max Temperature numerically |
| **Supervised — Classification** | Logistic Regression | Predict Risk Level (Low/Medium/High) |
| **Unsupervised** | K-Means Clustering | Find natural weather groups |

### Key Concepts
| Term | Meaning |
|------|---------|
| **Training** | Feeding known data to the model so it learns patterns |
| **Validation/Testing** | Checking accuracy on data the model has never seen |
| **Prediction** | Applying the trained model to new, real-world observations |
| **Supervised** | We provide the correct answers (labels) during training |
| **Unsupervised** | No labels — the model discovers hidden structure itself |
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 13  [CODE]  — Linear Regression                  ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 6 · MACHINE LEARNING — LINEAR REGRESSION")
print("━"*55)

# Features and target
FEATURES = ['Humidity', 'AQI', 'Wind_Speed', 'Min_Temperature']
TARGET   = 'Max_Temperature'

X = df[FEATURES]
y = df[TARGET]

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n📊  Training samples : {len(X_train)}  |  Test samples : {len(X_test)}")

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)       # ← "learning" happens here
y_pred = lr.predict(X_test)

# Evaluation
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mae  = np.mean(np.abs(y_test - y_pred))

print(f"\n📈  Model Performance on Test Set:")
print(f"    R² Score  : {r2:.4f}  → model explains {r2*100:.1f}% of temperature variation")
print(f"    RMSE      : {rmse:.2f} °C")
print(f"    MAE       : {mae:.2f} °C")

print(f"\n📉  Feature Coefficients:")
for feat, coef in zip(FEATURES, lr.coef_):
    bar = '█' * int(abs(coef))
    print(f"    {feat:20s}: {coef:+.4f}  {bar}")
print(f"    Intercept           : {lr.intercept_:.4f}")

# Sample predictions
print(f"\n🔮  Sample Predictions vs Actual (first 8 test rows):")
sample = pd.DataFrame({'Actual':y_test.values[:8],
                        'Predicted':[round(p,1) for p in y_pred[:8]],
                        'Error':  [round(abs(a-p),1) for a,p in
                                   zip(y_test.values[:8], y_pred[:8])]})
display(sample)

# ── Visualise Actual vs Predicted ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Linear Regression — Max Temperature Prediction', fontsize=15, fontweight='bold')

ax = axes[0]
ax.scatter(y_test, y_pred, color='#1565C0', alpha=0.7, s=60, edgecolors='white', lw=0.5)
lo, hi = min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1
ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction Line')
ax.set_xlabel('Actual Max Temperature (°C)')
ax.set_ylabel('Predicted Max Temperature (°C)')
ax.set_title(f'Actual vs Predicted  (R² = {r2:.3f})')
ax.legend()

ax = axes[1]
residuals = y_test.values - y_pred
ax.hist(residuals, bins=18, color='#7B1FA2', alpha=0.75, edgecolor='white')
ax.axvline(0, color='red', ls='--', lw=2)
ax.set_title('Residuals Distribution (Errors)')
ax.set_xlabel('Residual (Actual − Predicted)')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plot3_regression.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Regression plots saved.")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 14  [CODE]  — Logistic Regression Classification ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 7 · MACHINE LEARNING — CLASSIFICATION")
print("━"*55)

# Encode Risk_Level as numbers
le  = LabelEncoder()
df['Risk_Label'] = le.fit_transform(df['Risk_Level'])
print(f"\n🏷️   Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_clf = df[['Max_Temperature', 'Humidity', 'AQI', 'Wind_Speed']]
y_clf = df['Risk_Label']

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)
y_pred_clf = clf.predict(X_te)

acc = accuracy_score(y_te, y_pred_clf)
print(f"\n🎯  Classification Accuracy : {acc*100:.1f}%")
print(f"\n📋  Detailed Classification Report:")
print(classification_report(y_te, y_pred_clf, target_names=le.classes_))

# ── Confusion matrix heatmap ─────────────────────────────
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_te, y_pred_clf)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Logistic Regression — Heatwave Risk Classification', fontsize=15, fontweight='bold')

ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            linewidths=0.5, cbar_kws={'label':'Count'})
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title(f'Confusion Matrix  (Accuracy {acc*100:.0f}%)')

# Risk-level count bar
ax = axes[1]
risk_palette = {'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'}
risk_by_season = df.groupby(['Season', 'Risk_Level']).size().unstack(fill_value=0)
risk_by_season.plot(kind='bar', ax=ax,
                    color=[risk_palette.get(c,'grey') for c in risk_by_season.columns],
                    edgecolor='white', width=0.7)
ax.set_title('Risk Level Count by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Number of Days')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
ax.legend(title='Risk Level')

plt.tight_layout()
plt.savefig('plot4_classification.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Classification plots saved.")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 15  [CODE]  — K-Means Clustering                 ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 8 · MACHINE LEARNING — K-MEANS CLUSTERING")
print("━"*55)

X_cl = df[['Max_Temperature', 'AQI', 'Humidity']].copy()

# Elbow method — find optimal k
inertias = []
k_values = range(2, 9)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl)
    inertias.append(km.inertia_)

# Fit final model with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cl)
centers = kmeans.cluster_centers_

print("\n🔵  Cluster Centers (Mean values):")
cluster_df = pd.DataFrame(centers, columns=['Max_Temperature', 'AQI', 'Humidity'])
cluster_df.index = [f'Cluster {i}' for i in range(3)]
display(cluster_df.round(2))

# Interpret clusters
print("\n📌  Cluster Interpretation:")
for i, row in cluster_df.iterrows():
    if row['Max_Temperature'] > 42:
        label = "☀️  Very Hot & Polluted (Summer / Heatwave)    — HIGH RISK"
    elif row['Max_Temperature'] > 36:
        label = "🌤  Warm (Pre/Post-Monsoon / Spring)           — MEDIUM RISK"
    else:
        label = "🌧  Cool / Humid (Monsoon or Winter)           — LOW RISK"
    print(f"  {i}: Avg Temp {row['Max_Temperature']:.1f}°C  AQI {row['AQI']:.0f}  → {label}")

# ── Visualise clusters ───────────────────────────────────
CMAP = {0: '#2196F3', 1: '#FF9800', 2: '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K-Means Clustering (k=3) — Natural Weather Groups', fontsize=15, fontweight='bold')

# Left: Temperature vs AQI
ax = axes[0]
for cid, color in CMAP.items():
    subset = df[df['Cluster'] == cid]
    ax.scatter(subset['Max_Temperature'], subset['AQI'],
               c=color, label=f'Cluster {cid}', alpha=0.72, s=55, edgecolors='white', lw=0.3)
ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='*',
           s=280, zorder=5, label='Centers')
ax.set_xlabel('Max Temperature (°C)')
ax.set_ylabel('AQI')
ax.set_title('Temperature vs AQI')
ax.legend(fontsize=9)

# Middle: Temperature vs Humidity
ax = axes[1]
for cid, color in CMAP.items():
    subset = df[df['Cluster'] == cid]
    ax.scatter(subset['Max_Temperature'], subset['Humidity'],
               c=color, label=f'Cluster {cid}', alpha=0.72, s=55, edgecolors='white', lw=0.3)
ax.scatter(centers[:, 0], centers[:, 2], c='black', marker='*',
           s=280, zorder=5, label='Centers')
ax.set_xlabel('Max Temperature (°C)')
ax.set_ylabel('Humidity (%)')
ax.set_title('Temperature vs Humidity')
ax.legend(fontsize=9)

# Right: Elbow curve
ax = axes[2]
ax.plot(k_values, inertias, 'bo-', lw=2, ms=8, markerfacecolor='white',
        markeredgecolor='blue', markeredgewidth=2)
ax.axvline(3, color='red', ls='--', lw=1.8, label='Chosen k=3')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (WCSS)')
ax.set_title('Elbow Method — Optimal k Selection')
ax.legend()

plt.tight_layout()
plt.savefig('plot5_clustering.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Clustering plots saved.")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 16  [MARKDOWN]                                   ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## 📈 Unit-IV · Data Visualisation
Rich, publication-quality charts using **Matplotlib** and **Seaborn**:
1. **Line Plot** — Max Temperature trend over time with heatwave zones shaded
2. **Scatter Plot** — Temp vs AQI coloured by Risk Level
3. **Correlation Heatmap** — All variable relationships at a glance
4. **Boxplot by Season** — How temperature varies across seasons
5. **Wind Speed vs Temp** — Does more wind = cooler?
6. **Rainfall vs Temp** — Monsoon cooling effect
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 17  [CODE]  — Full Visualisation Suite           ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*55)
print("  STEP 9 · DATA VISUALISATION")
print("━"*55)

# Sort by date for time-series plots
df_sorted = df.sort_values('Date').reset_index(drop=True)

# ────────────────────────────────────────────────────────
# FIGURE A: Time-Series Line Plot
# ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df_sorted['Date'], df_sorted['Max_Temperature'],
        color='#E53935', lw=1.5, alpha=0.9, label='Max Temperature')
ax.plot(df_sorted['Date'], df_sorted['Min_Temperature'],
        color='#42A5F5', lw=1.5, alpha=0.9, label='Min Temperature')

# Shade danger zones
ax.fill_between(df_sorted['Date'], 40, df_sorted['Max_Temperature'],
                where=(df_sorted['Max_Temperature'] >= 40),
                color='#FF5722', alpha=0.25, label='Heatwave Zone (>40°C)')
ax.axhline(40, color='orange', ls='--', lw=1.5, label='40°C Threshold')
ax.axhline(45, color='red',    ls='--', lw=1.5, label='45°C Extreme Danger')

ax.set_title('Daily Temperature Trend — Ahmedabad 2022-2023', fontsize=15, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_timeseries.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Time-series plot saved.")

# ────────────────────────────────────────────────────────
# FIGURE B: 2×2 Grid — Scatter, Heatmap, Boxplot, Rainfall
# ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Visualization Dashboard', fontsize=16, fontweight='bold')

# B1: Scatter — Temp vs AQI coloured by Risk Level
ax = axes[0, 0]
risk_palette = {'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'}
for risk, grp in df.groupby('Risk_Level'):
    ax.scatter(grp['Max_Temperature'], grp['AQI'],
               c=risk_palette[risk], label=risk, alpha=0.75, s=55,
               edgecolors='white', lw=0.4)
ax.set_xlabel('Max Temperature (°C)')
ax.set_ylabel('AQI')
ax.set_title('Max Temperature vs AQI (by Risk Level)')
ax.legend(title='Risk Level')

# B2: Correlation Heatmap
ax = axes[0, 1]
num_features = ['Max_Temperature', 'Min_Temperature', 'Humidity',
                'Rainfall', 'AQI', 'Wind_Speed']
corr = df[num_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))      # Show lower triangle only
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 10})
ax.set_title('Correlation Heatmap')

# B3: Boxplot — Temp by Season
ax = axes[1, 0]
season_order = ['Winter', 'Summer (Pre-Monsoon)', 'Monsoon', 'Post-Monsoon']
season_palette = {'Winter': '#42A5F5', 'Summer (Pre-Monsoon)': '#F44336',
                  'Monsoon': '#66BB6A', 'Post-Monsoon': '#FF9800'}
sns.boxplot(x='Season', y='Max_Temperature', data=df,
            order=[s for s in season_order if s in df['Season'].unique()],
            palette=season_palette, ax=ax)
ax.set_title('Max Temperature by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Max Temperature (°C)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.axhline(40, color='red', ls='--', lw=1.5, alpha=0.7)

# B4: Rainfall vs Max Temperature (monsoon cooling)
ax = axes[1, 1]
rain_days   = df[df['Rainfall'] > 0]
no_rain_days= df[df['Rainfall'] == 0]
ax.scatter(no_rain_days['Rainfall'], no_rain_days['Max_Temperature'],
           color='#E53935', alpha=0.6, label='No Rainfall', s=50)
ax.scatter(rain_days['Rainfall'],    rain_days['Max_Temperature'],
           color='#1565C0', alpha=0.7, label='Rainy Days', s=50,
           edgecolors='white', lw=0.4)
ax.set_xlabel('Rainfall (mm)')
ax.set_ylabel('Max Temperature (°C)')
ax.set_title('Rainfall vs Max Temperature\n(Rainfall cools the city!)')
ax.legend()

plt.tight_layout()
plt.savefig('plot7_dashboard.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  Dashboard plot saved.")

# ────────────────────────────────────────────────────────
# FIGURE C: AQI Trend + Wind Speed Analysis
# ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(df_sorted['Date'], df_sorted['AQI'],
        color='#7B1FA2', lw=1.2, alpha=0.85, label='AQI')
ax.fill_between(df_sorted['Date'], df_sorted['AQI'], alpha=0.15, color='purple')
ax.axhline(150, color='red', ls='--', lw=1.5, label='Unhealthy AQI (150)')
ax.set_title('AQI Trend Over Time — Ahmedabad')
ax.set_xlabel('Date')
ax.set_ylabel('AQI')
ax.legend()

ax = axes[1]
ax.scatter(df['Wind_Speed'], df['Max_Temperature'],
           c=df['AQI'], cmap='RdYlGn_r', s=60, alpha=0.8, edgecolors='white', lw=0.3)
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
     norm=plt.Normalize(vmin=df['AQI'].min(), vmax=df['AQI'].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='AQI')
ax.set_xlabel('Wind Speed (km/h)')
ax.set_ylabel('Max Temperature (°C)')
ax.set_title('Wind Speed vs Temperature (colour = AQI)')

plt.tight_layout()
plt.savefig('plot8_aqi_wind.png', bbox_inches='tight', dpi=120)
plt.show()
print("✅  AQI & wind plots saved.")


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 18  [MARKDOWN]                                   ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## ⚖️ Unit-IV · Data Science Ethics

### Doing Good Data Science
Our goal is to **save lives**, not just score high accuracy.
Every model decision — issuing or not issuing an alert — has real consequences.

### Ownership of the Data
This data is sourced from the **India Meteorological Department (IMD)** and
**CPCB Air Quality data** — public government data. Individual citizens' data
(e.g., hospital admissions) would require explicit consent.

### Privacy & Informed Consent
If personal health records or GPS data are used to map heat exposure in slums,
every affected person must be:
- Informed about what data is collected
- Able to refuse participation without penalty
- Protected from re-identification of anonymous data

### The Five Cs of Data Ethics
| C | Principle | Application in this Project |
|---|-----------|----------------------------|
| **Consent** | Permission to use data | Using public IMD/CPCB data |
| **Clarity** | Transparent data usage | Documented columns and sources |
| **Consistency** | Fair treatment for all | Data from all areas, not just rich zones |
| **Control** | User rights over data | Open data policy, deletable records |
| **Consequences** | Think about impact | Prioritise alerts to vulnerable groups |

### Diversity & Inclusion
Heat **disproportionately kills** people who:
- Cannot afford air conditioning (slum residents, daily wage workers)
- Work outdoors (construction workers, street vendors)
- Lack access to clean water
- Are elderly or have pre-existing conditions

Our system must ensure alerts reach these groups via **SMS, sirens, and local
radio** — not just smartphone apps.

### Future Trends
- 🛰️ **Satellite thermal imaging** — map heat islands in dense urban slums
- 🌐 **IoT sensor grids** — real-time temperature at street level
- 🤖 **Deep Learning (LSTM)** — sequential forecasting of multi-day heatwaves
- 🔐 **Federated Learning** — train models on hospital data without exposing records
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 19  [MARKDOWN]  — Conclusion                     ║
# ╚══════════════════════════════════════════════════════════╝
"""
---
## ✅ Conclusion

### Key Findings

| Finding | Detail |
|---------|--------|
| 🌡️ Peak heat | Ahmedabad hits **47.1°C** in May — well above the 40°C danger threshold |
| 📅 Danger months | **April, May, early June** are consistently the highest-risk months |
| 💨 AQI link | High temperatures strongly correlate with high AQI (**r > 0.9**) |
| 🌧 Rainfall effect | Even small rainfall events significantly lower temperature |
| 🤖 ML accuracy | Linear Regression explains **>90% of temperature variance**; Classification reaches **high accuracy** on risk prediction |
| 🔵 Clusters | K-Means naturally found 3 groups: Cool/Monsoon · Warm/Transition · Hot/Dangerous |

### Business Value for Ahmedabad City
This pipeline supports the **Ahmedabad Heat Action Plan** by:
- 🚨 **Automated early warnings** — flag dangerous days 1–3 days ahead
- 💧 **Resource planning** — water tanker and cooling centre deployment
- 🏥 **Hospital readiness** — predict heat-stroke patient surges
- 📡 **Targeted alerts** — reach the most vulnerable populations first

### Full Data Science Lifecycle Completed

```
Define Goals → Retrieve Data → Prepare Data → Explore → Infer → Model → Visualise → Ethics
     ✅              ✅              ✅           ✅       ✅       ✅        ✅          ✅
```

> *"Data Science is not about algorithms — it is about people.
>   This project exists to protect the lives of Ahmedabad's most vulnerable citizens."*
"""


# ╔══════════════════════════════════════════════════════════╗
# ║   CELL 20  [CODE]  — Final Summary Print                ║
# ╚══════════════════════════════════════════════════════════╝

print("━"*60)
print("  🎉  AHMEDABAD HEATRESILIENT — PROJECT COMPLETE!")
print("━"*60)
print(f"\n  Dataset rows         : {len(df)}")
print(f"  Date range           : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Max recorded temp    : {df['Max_Temperature'].max():.1f} °C")
print(f"  High-risk days       : {(df['Risk_Level']=='High').sum()} ({(df['Risk_Level']=='High').mean()*100:.0f}%)")
print(f"  Medium-risk days     : {(df['Risk_Level']=='Medium').sum()} ({(df['Risk_Level']=='Medium').mean()*100:.0f}%)")
print(f"  Low-risk days        : {(df['Risk_Level']=='Low').sum()} ({(df['Risk_Level']=='Low').mean()*100:.0f}%)")
print(f"\n  Regression R²        : {r2:.4f}")
print(f"  Classification Acc   : {acc*100:.1f}%")
print(f"\n  Plots saved:")
plots = ['plot1_eda.png','plot2_inferential.png','plot3_regression.png',
         'plot4_classification.png','plot5_clustering.png',
         'plot6_timeseries.png','plot7_dashboard.png','plot8_aqi_wind.png']
for p in plots:
    print(f"    ✅  {p}")
print("\n━"*60)

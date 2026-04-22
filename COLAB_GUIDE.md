# 📒 Google Colab Setup Guide — Ahmedabad HeatResilient Project

## Step 1: Open Google Colab
- Go to: https://colab.research.google.com
- Click **New Notebook**

## Step 2: Upload Your Dataset
- In the left sidebar, click the 📁 **Files** icon
- Click **Upload** and upload `ahmedabad_heat_data.csv`

## Step 3: Notebook Cell Structure

Create cells in this order (alternate between Markdown ✍️ and Code 💻):

| # | Type | Content |
|---|------|---------|
| 1 | ✍️ Markdown | Project Title, Name, Objective (CELL 1) |
| 2 | 💻 Code | Import all libraries (CELL 2) |
| 3 | ✍️ Markdown | "Retrieving Data" heading (CELL 3) |
| 4 | 💻 Code | Load CSV, df.head(), df.shape (CELL 4) |
| 5 | ✍️ Markdown | "Data Preparation" heading (CELL 5) |
| 6 | 💻 Code | Missing values, Risk_Level column (CELL 6) |
| 7 | ✍️ Markdown | "Descriptive Statistics" heading (CELL 7) |
| 8 | 💻 Code | Statistics, mean, std, CV, skewness (CELL 8) |
| 9 | 💻 Code | Histogram and boxplot (CELL 9) |
| 10 | ✍️ Markdown | "Inferential Statistics" heading (CELL 10) |
| 11 | 💻 Code | T-test, normality, confidence interval (CELL 11) |
| 12 | ✍️ Markdown | "Machine Learning" heading (CELL 12) |
| 13 | 💻 Code | Linear Regression (CELL 13) |
| 14 | 💻 Code | Classification (CELL 14) |
| 15 | 💻 Code | K-Means Clustering (CELL 15) |
| 16 | ✍️ Markdown | "Data Visualization" heading (CELL 16) |
| 17 | 💻 Code | All 4 plots (CELL 17) |
| 18 | ✍️ Markdown | "Data Science Ethics" (CELL 18) |
| 19 | ✍️ Markdown | "Conclusion" (CELL 19) |

## Step 4: Add Markdown Cell
- Click **+ Text** (top-left) to add a Markdown cell
- Paste the Markdown content (text between triple quotes after [MARKDOWN])

## Step 5: Add Code Cell
- Click **+ Code** to add a code cell
- Paste the code content (between the triple quotes after [CODE])
- Remove the triple-quoted strings — just paste the raw code!

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `FileNotFoundError: ahmedabad_heat_data.csv` | Upload the CSV file in the Files panel first |
| `KeyError: 'AQI'` | Make sure your CSV has a column named exactly `AQI` |
| `ModuleNotFoundError` | The first cell (imports) has not been run yet — run it first |
| Plots not showing | Add `plt.show()` at the end (already included) |

## If Your CSV Has Different Column Names

Open your CSV and check the exact column names. Then in the code, replace:
- `Max_Temperature` → your actual temperature column name
- `AQI` → your actual AQI column name
- `Humidity` → your actual humidity column name

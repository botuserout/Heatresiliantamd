# 🚀 HOW TO RUN THIS PROJECT IN GOOGLE COLAB
# ================================================
# Recommended Platform: Google Colab  ✅ (FREE, no install needed)
# Alternative:          Jupyter Notebook (if you have Python installed locally)
# ================================================

## ✅ BEST OPTION — Google Colab (Recommended)

### Why Colab?
- Free GPU/TPU access
- No installation required
- Runs in your browser
- All libraries (pandas, sklearn, seaborn) are pre-installed

### Step-by-Step Instructions:

1. Open your browser → go to https://colab.research.google.com
2. Click "+ New Notebook"
3. Upload the dataset:
   - Look at the LEFT sidebar in Colab
   - Click the 📁 (Files) icon
   - Click "Upload to session storage"
   - Select `ahmedabad_heat_data.csv` from your computer
4. Open `ahmedabad_colab_notebook.py` in Notepad or VS Code
5. Follow the CELL markers to create cells:

   For a [MARKDOWN] cell → click "+ Text"  in the toolbar
   For a [CODE] cell    → click "+ Code"  in the toolbar

6. Copy-paste each cell's content accordingly
7. Run each cell with SHIFT+ENTER (top to bottom)

## 📋 Your 20-Cell Notebook Structure

| Cell | Type | Content |
|------|------|---------|
| 1 | ✍️ Text | Project title, objectives, dataset description |
| 2 | 💻 Code | Import all libraries |
| 3 | ✍️ Text | "Retrieving Data" section heading |
| 4 | 💻 Code | Load CSV, df.head(), df.info() |
| 5 | ✍️ Text | "Data Preparation" heading + Risk Level table |
| 6 | 💻 Code | Cleaning, datetime, Season, Risk_Level columns |
| 7 | ✍️ Text | "Descriptive Statistics" heading + concepts |
| 8 | 💻 Code | mean, median, std, CV, skewness, kurtosis |
| 9 | 💻 Code | Histogram + boxplot + monthly bar + risk pie |
| 10 | ✍️ Text | "Inferential Statistics" + CLT + Type I/II |
| 11 | 💻 Code | Shapiro-Wilk, T-test, Confidence Interval |
| 12 | ✍️ Text | "Machine Learning" heading + concepts table |
| 13 | 💻 Code | Linear Regression (predict max temp) |
| 14 | 💻 Code | Logistic Regression (classify risk level) |
| 15 | 💻 Code | K-Means Clustering + Elbow Method |
| 16 | ✍️ Text | "Data Visualization" heading |
| 17 | 💻 Code | 8 beautiful plots (line, scatter, heatmap, box…) |
| 18 | ✍️ Text | "Data Science Ethics" (Five Cs, diversity, etc.) |
| 19 | ✍️ Text | "Conclusion" with results table |
| 20 | 💻 Code | Final summary printout |

## 🐛 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `FileNotFoundError: ahmedabad_heat_data.csv` | Upload the CSV in Colab's Files panel first |
| `display() not defined` | Make sure you're in Colab; or replace with `print(df.head())` |
| Module not found | Run Cell 2 (imports) first before any other cell |
| Plot not showing | Already has `plt.show()` — just run the cell |

## 🖥️ ALTERNATIVE — Jupyter Notebook (Local)

If you prefer running locally:
1. Install Python + Jupyter: `pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn`
2. Open terminal → `jupyter notebook`
3. Create a new notebook in the same folder as `ahmedabad_heat_data.csv`
4. Replace `display(df)` with `df` (Jupyter renders DataFrames natively)
5. Run cells top to bottom

## 📊 Plots Generated (saved as PNG in the same folder)
- `plot1_eda.png`        — Histogram, boxplot, monthly bar, risk pie
- `plot2_inferential.png`— Normal curve, T-test boxplots
- `plot3_regression.png` — Actual vs Predicted, Residuals
- `plot4_classification.png` — Confusion matrix, seasonal risk bars
- `plot5_clustering.png` — 3 cluster scatter plots + Elbow curve
- `plot6_timeseries.png` — Temperature trend line with heatwave zones
- `plot7_dashboard.png`  — Scatter, correlation heatmap, season boxplot, rainfall
- `plot8_aqi_wind.png`   — AQI trend, wind vs temp

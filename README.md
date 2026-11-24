# Flight Ticket Price Prediction

A machine learning project aimed at predicting domestic flight ticket prices (`price`) based on airline, route, timing, stops, and booking lead time. The model performance is evaluated using **Root Mean Squared Error (RMSE)** and **$R^2$ Score**.

---

## 📊 Dataset Overview

The dataset is split into training and testing sets:
- **Training Data (`train.csv`):** 40,000 rows × 12 columns
- **Testing Data (`test.csv`):** 10,000 rows × 11 columns (excludes target variable `price`)

### Feature Dictionary

| Feature | Type | Description |
| :--- | :--- | :--- |
| `id` | Identifier | Unique record identifier (dropped before modeling) |
| `airline` | Categorical | Operating carrier (e.g., IndiGo, Vistara, Air India) |
| `flight` | Categorical | Flight code identifier (e.g., `UK-822`, `6E-2156`) |
| `class` | Categorical | Cabin tier (`Economy` or `Business`) |
| `source` | Categorical | Departure city |
| `destination` | Categorical | Arrival city |
| `departure` | Categorical | Time-of-day window for departure (e.g., `Morning`, `Evening`) |
| `arrival` | Categorical | Time-of-day window for arrival |
| `stops` | Categorical | Number of layovers (`zero`, `one`, `two_or_more`) |
| `duration` | Numerical | Total flight travel time in hours |
| `days_left` | Numerical | Days remaining between booking date and departure date |
| `price` | Numerical (Target) | Ticket price in INR (`₹`) |

---

## 🔍 Exploratory Data Analysis (EDA) Summary

### 1. Data Integrity & Cleaning
- **Hidden Missing Values:** Standardized placeholder strings (`""`, `" "`, `"?"`, `"NA"`, `"null"`) into `np.nan`.
- **Missing Value Distribution:** Detected missing entries across 5 features (`airline`, `departure`, `stops`, `duration`, `days_left`). Handled inside the preprocessing pipeline via median imputation (numerical) and mode imputation (categorical) to prevent data leakage.
- **Duplicates:** Verified **0 exact row duplicates**. Retained 488 feature-identical rows as legitimate independent passenger bookings on the same flight.
- **Outliers:** Flagged IQR upper outliers in `duration` (0.60%) and `price` (0.03%). All records were retained as valid domain cases (long multi-stop routes and Business Class fares).

### 2. Key Insights
- **Cabin Class Impact:** Cabin tier is the strongest primary driver of fare variance. **Business Class** tickets have a median price of **₹53,209**, compared to **₹5,660** for **Economy Class** (~9.4× difference).
- **Booking Lead Time (`days_left`):** Ticket prices exhibit non-linear pricing behavior as `days_left` approaches departure, with sharp price increases for last-minute bookings.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn` (Pipelines, `ColumnTransformer`, `SimpleImputer`, encoders, scalers, regression models)

---

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

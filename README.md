# Black-Friday-Sales-Prediction
Black Friday Sales Prediction App built with Streamlit. Upload training and test CSV files, select a model (Random Forest or Linear Regression), view insights, and predict purchase amounts. Includes data preprocessing, EDA, model evaluation, and downloadable results.



# 🛍️ Black Friday Sales Prediction

A Streamlit-powered web app that predicts customer purchase amounts during Black Friday sales using machine learning. Users can upload training and test datasets, choose a model, visualize data insights, and download the predicted results.

---

## 🚀 Features

- 📂 Upload your own `train.csv` and `test.csv`
- 📊 Automatic preprocessing and cleaning
- 📈 Visual insights: sales distribution, correlation heatmap
- 🤖 Select from **Random Forest** or **Linear Regression** models
- 📏 Model evaluation with RMSE
- 📋 View top 10 predictions
- 💾 Download full prediction as CSV

---

## 🛠️ Tools & Technologies Used

| Category | Tools |
|---------|-------|
| **Frontend/UI** | [Streamlit](https://streamlit.io/) |
| **Data Handling** | pandas, NumPy |
| **Visualization** | seaborn, matplotlib |
| **ML Models** | scikit-learn (RandomForestRegressor, LinearRegression) |
| **Metrics** | RMSE (Root Mean Squared Error) |

---

## 🧠 Algorithms Used

### 1. **Random Forest Regressor**
- Ensemble-based model using multiple decision trees.
- Handles non-linearity and interactions well.
- Robust against overfitting on large datasets.

### 2. **Linear Regression**
- Assumes linear relationship between features and target (`Purchase`).
- Simple, interpretable model suitable for quick testing.

---

## ⚙️ Preprocessing Steps

- Fill missing values in `Product_Category_2` and `Product_Category_3` with 0
- Encode categorical columns using **LabelEncoder**
  - `Gender`, `Age`, `City_Category`, `Stay_In_Current_City_Years`
- Drop `User_ID` and `Product_ID` before training
- Train-test split for model evaluation

---

## 📂 Dataset Info

- Based on Black Friday dataset from a retail store.
- Sample columns:
  - `User_ID`, `Product_ID`, `Gender`, `Age`, `City_Category`, `Purchase`, etc.

---

## 📦 How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py

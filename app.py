import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(page_title="Black Friday Predictor", layout="wide")

# 🎞️ Welcome Animation
st.image("https://media.giphy.com/media/Rk5KD9r3WZkXqG2Jrk/giphy.gif", use_column_width=True)
st.title("🛍️ Black Friday Sales Prediction Dashboard")

st.markdown("Welcome to your own smart prediction system! Let’s visualize, model, and predict Black Friday trends 🎯")

# File Upload
train_file = st.file_uploader("📂 Upload your train.csv", type="csv")
test_file = st.file_uploader("📂 Upload your test.csv", type="csv")

# Model Selector
model_name = st.sidebar.selectbox("Choose Prediction Model", ["Random Forest", "Linear Regression"])

# Preprocessing
def preprocess(df):
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    le = LabelEncoder()
    for col in ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']:
        df[col] = le.fit_transform(df[col])
    df['Marital_Status'] = df['Marital_Status'].astype(int)
    return df

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    st.success("✅ Files uploaded successfully!")

    # Preprocess
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # 📊 Display Key Stats
    st.subheader("📈 Snapshot of Training Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧑‍💼 Total Customers", f"{train_df['User_ID'].nunique()}")
    col2.metric("💰 Total Purchase ₹", f"{int(train_df['Purchase'].sum()):,}")
    top_age = train_df['Age'].mode()[0]
    col3.metric("🎯 Top Age Group", top_age)

    # 📊 Sales Distribution
    st.subheader("🪄 Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(train_df['Purchase'], bins=50, ax=ax)
    st.pyplot(fig)

    # 🔗 Correlation
    st.subheader("🔗 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    numeric_data = train_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", ax=ax2, cmap="YlGnBu")
    st.pyplot(fig2)

    # Training the Model
    st.subheader("⚙️ Model Training & Prediction")

    X_train = train_df.drop(['User_ID', 'Product_ID', 'Purchase'], axis=1)
    y_train = train_df['Purchase']
    X_test = test_df.drop(['User_ID', 'Product_ID'], axis=1)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    # Train and Validate
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    st.info(f"📏 RMSE (Validation): {rmse:.2f}")
    y_pred = model.predict(X_test)

    # Output Preview
    st.subheader("📋 Sample Predictions")
    submission = test_df[['User_ID', 'Product_ID']].copy()
    submission['Predicted_Purchase'] = y_pred
    st.dataframe(submission.head(10))

    # 📥 Download Button
    csv = submission.to_csv(index=False)
    st.download_button("📥 Download Submission CSV", data=csv, file_name="submission.csv", mime="text/csv")

    # 🧠 Future Forecast (Fun)
    st.markdown("### 🔮 Future Insight (Mock Forecast)")
    st.success("📊 Expect a 12% increase in average purchase next Black Friday!")

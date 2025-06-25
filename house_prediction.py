import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App (Region + BHK + Size)")

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)

        st.write("✅ **Dataset Preview**")
        st.dataframe(df.head())

        required_cols = ['BHK', 'Size (sqft)', 'Region_Type', 'Total_Price']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns: " + ", ".join(required_cols))
        else:
            X = df[['BHK', 'Size (sqft)', 'Region_Type']]
            y = df['Total_Price']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Region_Type'])
                ],
                remainder='passthrough'
            )

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            st.write("📈 **Model Performance**")
            st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
            st.write("R² Score:", round(r2_score(y_test, y_pred), 4))

            st.sidebar.header("🏗️ Enter House Details")

            input_region = st.sidebar.selectbox("Select Region", df['Region_Type'].unique())
            input_bhk = st.sidebar.slider("Number of BHK", min_value=1, max_value=4, value=3)
            input_size = st.sidebar.number_input("Size in sqft", min_value=300, max_value=10000, value=1200)

            input_data = pd.DataFrame([{
                'BHK': input_bhk,
                'Size (sqft)': input_size,
                'Region_Type': input_region
            }])

            predicted_price = model.predict(input_data)[0]
            predicted_price = max(100000, predicted_price)

            st.sidebar.subheader("💰 Predicted Price")
            st.sidebar.success(f"₹ {predicted_price:,.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

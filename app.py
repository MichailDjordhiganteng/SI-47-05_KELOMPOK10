import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load("model_regresi.pkl")
scaler = joblib.load("scaler.pkl")


# Load data
df = pd.read_csv('Online Retail.csv')
df = df.drop_duplicates()
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Hitung snapshot date dan RFM
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# Scoring
rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm['Segment'] = rfm['RFM_Score'].apply(lambda x: 'Best' if x == '444' else 'Others')

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Prediksi Sales", "Visualisasi RFM"])

if page == "Prediksi Sales":
    st.title("Prediksi Sales")
    qty = st.number_input("Masukkan Quantity", min_value=1)
    price = st.number_input("Masukkan Unit Price", min_value=0.01, format="%.2f")

    if st.button("Prediksi"):
        input_data = np.array([[qty, price]])
        input_scaled = scaler.transform(input_data)
        result = model.predict(input_scaled)[0]
        st.success(f"Prediksi Sales: £{result:.2f}")

# Halaman Visualisasi
elif page == "Visualisasi RFM":
    st.title("Visualisasi RFM")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    sns.histplot(rfm['Recency'], bins=30, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Histogram Recency')
    sns.histplot(rfm['Frequency'], bins=30, ax=axes[1, 0], kde=True)
    axes[1, 0].set_title('Histogram Frequency')
    sns.histplot(rfm['Monetary'], bins=30, ax=axes[2, 0], kde=True)
    axes[2, 0].set_title('Histogram Monetary')

    sns.boxplot(x=rfm['Recency'], ax=axes[0, 1], color='lightblue')
    axes[0, 1].set_title('Boxplot Recency')
    sns.boxplot(x=rfm['Frequency'], ax=axes[1, 1], color='lightgreen')
    axes[1, 1].set_title('Boxplot Frequency')
    sns.boxplot(x=rfm['Monetary'], ax=axes[2, 1], color='lightcoral')
    axes[2, 1].set_title('Boxplot Monetary')
    st.pyplot(fig)

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi RFM")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Segmentasi
    st.subheader("Distribusi Segment")
    st.bar_chart(rfm['Segment'].value_counts())


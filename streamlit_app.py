import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('Online Retail.csv', encoding='ISO-8859-1')
df.columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
              'UnitPrice', 'CustomerID', 'Country']

# Data preparation
df = df.drop_duplicates()
df = df.dropna(subset=['CustomerID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# RFM Scoring
rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

# Target & Model
rfm['Target'] = (rfm['Monetary'] > 1000).astype(int)
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Streamlit App
st.title("📊 Dashboard Segmentasi & Prediksi Pelanggan Retail")

# Input Customer ID
input_id = st.sidebar.number_input("Masukkan Customer ID", 
min_value=int(rfm['CustomerID'].min()), 
max_value=int(rfm['CustomerID'].max()), step=1)

if input_id in rfm['CustomerID'].astype(int).values:
    selected = rfm[rfm['CustomerID'].astype(int) == input_id]
    
    st.subheader("📌 Detail Pelanggan")
    st.write(selected[['Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Cluster']])
    
    # Probabilitas pembelian ulang
    input_scaled = scaler.transform(selected[['Recency', 'Frequency', 'Monetary']])
    proba = model.predict_proba(input_scaled)[0][1]
    st.metric("🎯 Probabilitas Akan Membeli Ulang", f"{proba:.2%}")

    # Filter berdasarkan cluster pelanggan
    selected_cluster = selected['Cluster'].values[0]
    filtered_rfm = rfm[rfm['Cluster'] == selected_cluster]

    # Distribusi RFM (khusus cluster)
    st.subheader("📈 Distribusi RFM (Cluster yang Sama)")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_rfm['Recency'], bins=30, kde=True, ax=ax1)
        ax1.set_title("Distribusi Recency (Cluster)")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_rfm['Monetary'], bins=30, kde=True, ax=ax2)
        ax2.set_title("Distribusi Monetary (Cluster)")
        st.pyplot(fig2)

    # Korelasi antar variabel
    st.subheader("🔥 Korelasi Variabel RFM (Cluster yang Sama)")
    fig3, ax3 = plt.subplots()
    sns.heatmap(filtered_rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Ringkasan cluster pelanggan yang sama
    st.subheader("📊 Karakteristik Cluster Saat Ini")
    summary = filtered_rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    st.dataframe(summary)

    # Distribusi RFM Score dalam cluster
    st.subheader("📉 Distribusi Skor RFM (Cluster yang Sama)")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=filtered_rfm, x='RFM_Score', order=filtered_rfm['RFM_Score'].value_counts().index[:20], ax=ax4)
    ax4.set_title("Sebaran Skor RFM (Cluster)")
    ax4.tick_params(axis='x', rotation=90)
    st.pyplot(fig4)

else:
    st.warning("Customer ID tidak ditemukan dalam data.")

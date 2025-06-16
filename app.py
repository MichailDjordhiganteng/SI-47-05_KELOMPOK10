import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

# Set halaman config
st.set_page_config(page_title="RFM App", layout="wide")

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("Online Retail.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

# Data dan RFM
df = load_data()
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]
rfm['Segment'] = rfm.apply(lambda x: 'Best' if x['Recency'] <= rfm['Recency'].quantile(0.25)
                                        and x['Frequency'] >= rfm['Frequency'].quantile(0.75)
                                        and x['Monetary'] >= rfm['Monetary'].quantile(0.75) else 'Others', axis=1)

# Navigasi dengan option_menu
selected = option_menu(
    menu_title=None,
    options=["Train Model", "Predict", "Clustering"],
    icons=["bar-chart-line", "search", "diagram-3"],
    orientation="horizontal"
)

# Train Model
if selected == "Train Model":
    st.title("🧠 Training Model Klasifikasi Pelanggan")

    split_option = st.selectbox("Pilih Split Rasio", ["70:30", "80:20", "65:35", "75:25"])
    split_dict = {"70:30": 0.3, "80:20": 0.2, "65:35": 0.35, "75:25": 0.25}
    test_size = split_dict[split_option]

    rfm_train = rfm.copy()
    rfm_train['SegmentLabel'] = rfm_train['Segment'].apply(lambda x: 1 if x == 'Best' else 0)
    X = rfm_train[['Recency', 'Frequency', 'Monetary']]
    y = rfm_train['SegmentLabel']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, f"model_{split_option.replace(':','_')}.pkl")
    joblib.dump(scaler, f"scaler_{split_option.replace(':','_')}.pkl")

    st.success("Model dan Scaler disimpan.")
    st.metric("Akurasi Training", f"{model.score(X_train, y_train):.2f}")
    st.metric("Akurasi Testing", f"{model.score(X_test, y_test):.2f}")

    cm = confusion_matrix(y_test, model.predict(X_test))
    st.write("### Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Predict
elif selected == "Predict":
    st.title("🔍 Prediksi Segment Pelanggan")
    split_option = st.selectbox("Pilih Split yang Digunakan", ["70:30", "80:20", "65:35", "75:25"])
    try:
        model = joblib.load(f"model_{split_option.replace(':','_')}.pkl")
        scaler = joblib.load(f"scaler_{split_option.replace(':','_')}.pkl")

        rec = st.number_input("Recency", min_value=0)
        freq = st.number_input("Frequency", min_value=0)
        mon = st.number_input("Monetary", min_value=0.0)

        if st.button("Prediksi Segment"):
            input_data = scaler.transform([[rec, freq, mon]])
            result = model.predict(input_data)[0]
            label = "Best" if result == 1 else "Others"
            st.success(f"Segment Pelanggan: {label}")
    except:
        st.warning("Model belum dilatih untuk rasio tersebut.")

# Clustering
elif selected == "Clustering":
    st.title("📊 Clustering Pelanggan dengan K-Means")
    cluster_option = st.selectbox("Pilih Jumlah Cluster", [3, 4, 5, 6])

    X = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=cluster_option, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write("### Distribusi Pelanggan per Cluster")
    st.bar_chart(rfm['Cluster'].value_counts().sort_index())

    st.write("### Visualisasi Cluster (2D)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='tab10')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Monetary')
    ax.set_title('Recency vs Monetary by Cluster')
    st.pyplot(fig)
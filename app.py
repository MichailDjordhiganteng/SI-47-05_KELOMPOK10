import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

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

df = load_data()

# Hitung RFM
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# Segmentasi untuk klasifikasi
rfm['Segment'] = rfm.apply(lambda x: 'Best' if x['Recency'] <= rfm['Recency'].quantile(0.25) and
                                          x['Frequency'] >= rfm['Frequency'].quantile(0.75) and
                                          x['Monetary'] >= rfm['Monetary'].quantile(0.75) else 'Others', axis=1)

# Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Training", "Validasi", "Prediksi Sales", "Visualisasi RFM", "Clustering"])

# Dashboard
if page == "Dashboard":
    st.title("📊 Dashboard RFM Analysis")
    st.metric("Jumlah Customer", rfm.shape[0])
    st.metric("Customer 'Best'", rfm[rfm['Segment'] == 'Best'].shape[0])
    fig, ax = plt.subplots()
    sns.histplot(rfm['Monetary'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Training
elif page == "Training":
    st.title("🧠 Training Model Klasifikasi")
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

    model_filename = f"model_{split_option.replace(':','_')}.pkl"
    scaler_filename = f"scaler_{split_option.replace(':','_')}.pkl"
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    st.success(f"Model disimpan untuk rasio {split_option}.")
    st.metric("Akurasi Training", f"{model.score(X_train, y_train):.2f}")
    st.metric("Akurasi Testing", f"{model.score(X_test, y_test):.2f}")
    cm = confusion_matrix(y_test, model.predict(X_test))
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Validasi
elif page == "Validasi":
    st.title("📈 Validasi Model")
    split_ratio = st.selectbox("Pilih Split", ["70:30", "80:20", "65:35", "75:25"])
    model_file = f"model_{split_ratio.replace(':','_')}.pkl"
    scaler_file = f"scaler_{split_ratio.replace(':','_')}.pkl"
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        X = rfm[['Recency', 'Frequency', 'Monetary']]
        y = rfm['Segment'].apply(lambda x: 1 if x == 'Best' else 0)
        test_size = float(split_ratio.split(':')[1])/100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi", f"{acc*100:.2f}%")
        cm = confusion_matrix(y_test, y_pred)
        st.text("Confusion Matrix:")
        st.text(cm)
    except:
        st.warning("Model belum tersedia. Silakan lakukan training terlebih dahulu.")

# Prediksi Sales
elif page == "Prediksi Sales":
    st.title("Prediksi Sales")
    qty = st.number_input("Masukkan Quantity", min_value=1)
    price = st.number_input("Masukkan Unit Price", min_value=0.01, format="%.2f")

    if st.button("Latih Model Regresi"):
        reg_data = df[['Quantity', 'UnitPrice', 'TotalPrice']]
        X = reg_data[['Quantity', 'UnitPrice']]
        y = reg_data['TotalPrice']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        joblib.dump(model, "model_regresi.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model regresi dilatih dan disimpan!")

    if st.button("Prediksi"):
        try:
            model = joblib.load("model_regresi.pkl")
            scaler = joblib.load("scaler.pkl")
            input_data = scaler.transform(np.array([[qty, price]]))
            result = model.predict(input_data)[0]
            st.success(f"Prediksi Sales: £{result:.2f}")
        except:
            st.warning("Model regresi belum tersedia. Silakan latih terlebih dahulu.")

# Visualisasi RFM
elif page == "Visualisasi RFM":
    st.title("Visualisasi RFM")
    rfm['RFM_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(str) + \
                       pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(str) + \
                       pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(str)

    st.bar_chart(rfm['Segment'].value_counts())
    fig, ax = plt.subplots()
    sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Clustering
elif page == "Clustering":
    st.title("🔍 Clustering Pelanggan (Unsupervised Learning)")
    k = st.slider("Pilih Jumlah Cluster", 2, 6, 3)
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Distribusi Cluster")
    st.bar_chart(rfm['Cluster'].value_counts())
    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

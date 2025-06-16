import streamlit as st
import pandas as pd
import numpy as np
<<<<<<< HEAD
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load("model_regresi.pkl")
scaler = joblib.load("scaler.pkl")

=======
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
>>>>>>> 9ca2670 (Tubes DATMIN)

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
@st.cache_data
def train_regression_model():
    df = load_data()
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Gunakan Quantity dan UnitPrice untuk memprediksi TotalPrice
    X = df[['Quantity', 'UnitPrice']]
    y = df['TotalPrice']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Simpan model & scaler
    joblib.dump(model, "model_regresi.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return model, scaler

# Jalankan fungsi ini sekali untuk buat file-nya
train_regression_model()

# Hitung RFM
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# Tambahkan Segment
rfm['Segment'] = rfm.apply(lambda x: 'Best' if x['Recency'] <= rfm['Recency'].quantile(0.25) and
                                          x['Frequency'] >= rfm['Frequency'].quantile(0.75) and
                                          x['Monetary'] >= rfm['Monetary'].quantile(0.75) else 'Others', axis=1)

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Dashboard", "Training", "Validasi", "Prediksi Sales", "Visualisasi RFM", "Clustering"])

<<<<<<< HEAD
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
=======
# Halaman Dashboard
if page == "Dashboard":
    st.title("\U0001F4CA Dashboard RFM Analysis")
    st.metric("Jumlah Customer", rfm.shape[0])
    st.metric("Customer 'Best'", rfm[rfm['Segment'] == 'Best'].shape[0])
    fig, ax = plt.subplots()
    sns.histplot(rfm['Monetary'], bins=30, kde=True, ax=ax)
>>>>>>> 9ca2670 (Tubes DATMIN)
    st.pyplot(fig)

# Halaman Training
elif page == "Training":
    st.title("\U0001F9E0 Training Model Klasifikasi")

    st.write("""
    Halaman ini digunakan untuk **melatih model klasifikasi** berdasarkan segmentasi pelanggan RFM.
    Model akan mempelajari pola pelanggan yang termasuk dalam kategori **'Best'** atau **'Others'**.
    """)

    split_option = st.selectbox("Pilih Split Rasio", ["70:30", "80:20", "65:35", "75:25"])

    split_dict = {
        "70:30": 0.3,
        "80:20": 0.2,
        "65:35": 0.35,
        "75:25": 0.25
    }

    test_size = split_dict[split_option]

    # Persiapkan fitur dan label
    rfm_train = rfm.copy()
    rfm_train['SegmentLabel'] = rfm_train['Segment'].apply(lambda x: 1 if x == 'Best' else 0)

    X = rfm_train[['Recency', 'Frequency', 'Monetary']]
    y = rfm_train['SegmentLabel']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Simpan
    model_filename = f"model_{split_option.replace(':','_')}.pkl"
    scaler_filename = f"scaler_{split_option.replace(':','_')}.pkl"
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    st.success(f"✅ Model dan Scaler disimpan untuk rasio split {split_option}.")
    st.info(f"Jumlah data training: {X_train.shape[0]} | Jumlah data testing: {X_test.shape[0]}")

    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    st.metric("Akurasi Training", f"{acc_train:.2f}")
    st.metric("Akurasi Testing", f"{acc_test:.2f}")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix (Data Testing)")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Halaman Validasi
elif page == "Validasi":
    st.title("📈 Validasi Model")
    split_ratio = st.selectbox("Pilih Split untuk Validasi", ["70:30", "80:20", "65:35", "75:25"])
    
    model_filename = f"model_{split_ratio.replace(':', '_')}.pkl"
    scaler_filename = f"scaler_{split_ratio.replace(':', '_')}.pkl"

    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        # Siapkan data
        X = rfm[['Recency', 'Frequency', 'Monetary']]
        y = rfm['Segment'].apply(lambda x: 1 if x == 'Best' else 0)
        
        test_size = float(split_ratio.split(':')[1]) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.metric("Akurasi", f"{acc * 100:.2f}%")
        st.text("Confusion Matrix:")
        st.text(cm)

    except FileNotFoundError:
        st.error(f"Model untuk split {split_ratio} belum dilatih. Silakan lakukan training terlebih dahulu.")


# Halaman Prediksi Sales
elif page == "Prediksi Sales":
    st.title("Prediksi Sales")
    st.warning("Halaman ini belum diaktifkan karena model regresi belum tersedia.")

# Halaman Visualisasi RFM
elif page == "Visualisasi RFM":
    st.title("Visualisasi RFM")
    rfm['RFM_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(str) + \
                       pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(str) + \
                       pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(str)

    label_map = {'1': 'J', '2': 'C', '3': 'S', '4': 'T'}
    rfm['RFM_Label'] = 'R:' + rfm['RFM_Score'].str[0].map(label_map) + \
                       ' F:' + rfm['RFM_Score'].str[1].map(label_map) + \
                       ' M:' + rfm['RFM_Score'].str[2].map(label_map)

    st.subheader("Sebaran Pelanggan berdasarkan Label RFM")
    st.bar_chart(rfm['RFM_Label'].value_counts())

    st.subheader("Heatmap Korelasi RFM")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Distribusi Segment")
    st.bar_chart(rfm['Segment'].value_counts())

# Halaman Clustering
elif page == "Clustering":
    st.title("\U0001F9EC Clustering Pelanggan (KMeans)")
    st.write("""
    Halaman ini melakukan **segmentasi pelanggan menggunakan metode KMeans Clustering**.
    Anda dapat memilih jumlah cluster dan melihat visualisasi hasil klasterisasi berdasarkan nilai RFM.
    """)

    cluster_option = st.selectbox("Pilih Jumlah Cluster", [3, 4, 5, 6])
    rfm_clustering = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clustering)

    kmeans = KMeans(n_clusters=cluster_option, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Distribusi Jumlah Pelanggan per Cluster")
    st.bar_chart(rfm['Cluster'].value_counts().sort_index())

    st.subheader("Visualisasi Cluster (Recency vs Frequency)")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster', palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Visualisasi Cluster (Monetary vs Frequency)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=rfm, x='Monetary', y='Frequency', hue='Cluster', palette='Set2', ax=ax2)
    st.pyplot(fig2)

    st.info("Silakan ubah jumlah cluster untuk melihat segmentasi yang berbeda.")

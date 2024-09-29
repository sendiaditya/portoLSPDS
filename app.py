import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Data Top Rank University")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df

    st.write("### File Uploaded Successfully")

    if st.button("Data Understanding"):
        st.write("### Data Understanding")
        st.write("5 baris pertama dataset:")
        st.write(df.head())
        st.write("5 baris terakhir dataset:")
        st.write(df.tail())
        st.write("Jumlah Baris: ", df.shape[0])
        st.write("Jumlah Kolom: ", df.shape[1])
        st.write("Tipe Data Tiap Kolom:")
        st.write(df.dtypes)
        st.write("Statistical Summary:")
        st.write(df.describe(include='all'))

        q1 = df.select_dtypes(exclude=['object']).quantile(0.25)
        q3 = df.select_dtypes(exclude=['object']).quantile(0.75)
        iqr = q3 - q1
        batas_bawah = q1 - (1.5 * iqr)
        batas_atas = q3 + (1.5 * iqr)
        st.write("IQR Outlier Boundaries:")
        st.write("Lower Bound:", batas_bawah)
        st.write("Upper Bound:", batas_atas)

        st.write("Histograms Tiap Kolom Numerik:")
        for column in df.select_dtypes(exclude=['object']):
            fig, ax = plt.subplots()
            ax.hist(df[column], bins=20)
            ax.set_title(column)
            st.pyplot(fig)

        st.write("Boxplots Tiap Kolom Numerik:")
        for column in df.select_dtypes(exclude=['object']):
            fig, ax = plt.subplots(figsize=(20, 2))
            sns.boxplot(data=df, x=column, ax=ax)
            st.pyplot(fig)

        st.write("Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(exclude="object").corr(), annot=True, ax=ax)
        st.pyplot(fig)

    if st.button("Data Preparation"):
        st.write("### Data Preparation")
        st.write("Missing Values Tiap kolom:")
        st.write(df.isna().sum())

        if 'N&S' in df.columns:
            imputer = KNNImputer(n_neighbors=2)
            df[['N&S']] = imputer.fit_transform(df[['N&S']])

        st.write("Missing Values Stelah Dilakukan Imputasi:")
        st.write(df.isna().sum())

        st.write("Outlier Di Handle Dengan Winsorizing:")
        columns_to_winsorize = ['Hici', 'N&S', 'PUB', 'PCP']
        for column in columns_to_winsorize:
            if column in df.columns:
                df[column] = winsorize(df[column], limits=[0, 0.05])

        st.write("Boxplots Setelah Dilakukannya Winsorizing:")
        for column in df.select_dtypes(exclude=['object']):
            fig, ax = plt.subplots(figsize=(20, 2))
            sns.boxplot(data=df, x=column, ax=ax)
            st.pyplot(fig)

        st.write("Heatmap Setelah Data Preparation:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(exclude="object").corr(), annot=True, ax=ax)
        st.pyplot(fig)

        st.session_state.df = df

    if st.button("Model"):
        st.write("### Model")
        df = st.session_state.df

        if df.isna().sum().sum() > 0:
            st.write("Masih ada missing values di dataset. Coba Lakukan Data Preparation terlebih dahulu dengan mengklik button Data Preparation.")
        else:
            dfknn = df.copy()
            scaler = RobustScaler()
            dfknn[['Alumni', 'Award']] = scaler.fit_transform(dfknn[['Alumni', 'Award']])

            cat_col = ['Rank', 'University_Name', 'National/Regional Rank']
            from sklearn import preprocessing
            le = preprocessing.LabelEncoder()
            dfknn[cat_col] = dfknn[cat_col].apply(le.fit_transform)

            X = dfknn[['Alumni', 'Award', 'Hici', 'N&S', 'PUB', 'PCP']].values
            y = dfknn['Rank'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            k = KFold(n_splits=5)
            score = cross_val_score(knn, X_train, y_train, scoring='accuracy', cv=k).mean()
            st.write(f"Akurasi Menggunakan Training Set: {round(score, 3)}")

            y_pred = knn.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Akurasi Menggunakan Test Set: {round(test_accuracy, 3)}")

            accuracy = []
            for i in range(1, 15):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(X_train, y_train)
                pred_i = knn.predict(X_test)
                accuracy_i = accuracy_score(y_test, pred_i)
                accuracy.append(accuracy_i)

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(range(1, 15), accuracy, color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            ax.set_title('Accuracy vs. K Value')
            ax.set_xlabel('K')
            ax.set_ylabel('Accuracy')
            st.pyplot(fig)

            knn = KNeighborsClassifier(n_neighbors=4)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Akurasi Akhir Menggunakan Test Set Dengan K=4: {round(final_accuracy * 100, 2)}%")

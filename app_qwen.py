import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import subprocess
import sys

# Configuración inicial
st.set_page_config(page_title="AutoML Predictor", layout="wide")
st.title("🔍 Predictive Model Builder from Kaggle or Upload")

# -------------------------
# Función para descargar dataset de Kaggle
# -------------------------
def download_kaggle_dataset(dataset_name):
    try:
        # Crear directorio temporal
        os.makedirs("kaggle_data", exist_ok=True)
        # Descargar con kaggle CLI
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_name, "-p", "kaggle_data", "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Error descargando dataset: {result.stderr}")
            return None
        
        # Buscar archivo CSV o Excel
        files = [f for f in os.listdir("kaggle_data") if f.endswith(('.csv', '.xlsx', '.xls'))]
        if not files:
            st.error("No se encontró archivo compatible en el dataset.")
            return None
        
        # Cargar primer archivo válido
        file_path = os.path.join("kaggle_data", files[0])
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -------------------------
# Función para entrenar modelo y graficar
# -------------------------
def build_and_predict(df, target_col):
    # Eliminar filas con NaN en target
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Codificar variables categóricas en X
    le_dict = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Codificar y si es categórico
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        is_classification = True
    else:
        is_classification = False
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Mostrar métricas
    st.subheader("📊 Resultados del modelo")
    if is_classification:
        st.text("Clasificación Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)
    else:
        st.write(f"R²: {r2_score(y_test, y_pred):.3f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        
        # Gráfico de predicción vs real
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Predicción vs Real")
        st.pyplot(fig)
    
    # Importancia de características
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = X.columns[indices][:10]
    
    fig, ax = plt.subplots()
    ax.barh(range(len(top_features)), importances[indices][:10])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_title("Top 10 Características Importantes")
    st.pyplot(fig)

# -------------------------
# Interfaz principal
# -------------------------
st.sidebar.header("📤 Cargar datos")

option = st.sidebar.radio("Fuente de datos", ("Subir archivo", "Dataset de Kaggle"))

df = None

if option == "Subir archivo":
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo (CSV o Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

elif option == "Dataset de Kaggle":
    dataset_name = st.sidebar.text_input("Nombre del dataset (ej: zhangliao/used-car-prices)")
    if dataset_name:
        with st.spinner("Descargando dataset de Kaggle..."):
            df = download_kaggle_dataset(dataset_name)

# Mostrar datos si existen
if df is not None:
    st.subheader("🔍 Vista previa de los datos")
    st.write(df.head())
    
    # Inferir tipos
    st.subheader("📋 Tipos de datos inferidos")
    dtype_info = pd.DataFrame({
        'Columna': df.columns,
        'Tipo Pandas': df.dtypes,
        'Valores únicos': df.nunique(),
        'Valores nulos': df.isnull().sum()
    })
    st.dataframe(dtype_info)
    
    # Seleccionar variable objetivo
    target_col = st.selectbox("Selecciona la variable objetivo", df.columns)
    
    if st.button("🚀 Entrenar modelo predictivo"):
        if df[target_col].isnull().all():
            st.error("La columna seleccionada está completamente vacía.")
        else:
            build_and_predict(df.copy(), target_col)
else:
    st.info("Por favor, sube un archivo o ingresa un dataset de Kaggle para comenzar.")

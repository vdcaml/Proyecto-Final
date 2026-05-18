
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io


# -------------------------------------------------------------------------------------------------------
# --- Clase 
# -------------------------------------------------------------------------------------------------------
class DataAnalyzer:

    def __init__(self, df):
        self.df = df
        self.var_numerica = df.select_dtypes(include=['int64','float64']).columns
        self.var_categorica = df.select_dtypes(include=['object']).columns

    def estadisticas_descriptivas(self, col):
        return {
            "media": round(self.df[col].mean(), 2),
            "mediana": round(self.df[col].median(), 2),
            "desviacion": round(self.df[col].std(), 2),
            "minimo": self.df[col].min(),
            "maximo": self.df[col].max()
        }

    def clasificar_variables(self):
        return list(self.var_numerica), list(self.var_categorica)

    def plot_histograma(self, col, bins=20, kde=True):
        fig, ax = plt.subplots()
        sns.histplot(self.df[col], bins=bins, kde=kde, ax=ax, color="skyblue")
        ax.set_title(f"Histograma de {col}")
        return fig

    def plot_boxplot(self, num_col, cat_col):
        fig, ax = plt.subplots()
        sns.boxplot(x=self.df[cat_col], y=self.df[num_col], ax=ax, palette="Set2")
        ax.set_title(f"{num_col} vs {cat_col}")
        plt.xticks(rotation=45)
        return fig

    def plot_barras(self, col):
        conteo = self.df[col].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=conteo.index, y=conteo.values, ax=ax, palette="pastel")
        plt.xticks(rotation=45)
        ax.set_title(f"Distribución de {col}")
        return fig

    def valores_faltantes(self):
        """Devuelve conteo de valores nulos por columna"""
        return self.df.isnull().sum()
    
    def tabla_contingencia(self, col1, col2, limit=20):
        """Devuelve tabla de contingencia entre dos variables categóricas"""
        tabla = pd.crosstab(self.df[col1], self.df[col2])
        return tabla.head(limit)

    def plot_contingencia(self, col1, col2):
        """Devuelve gráfico de barras apiladas normalizadas"""
        tabla_norm = pd.crosstab(self.df[col1], self.df[col2], normalize='index')
        fig, ax = plt.subplots(figsize=(8, 5))
        tabla_norm.plot(kind='bar', stacked=True, colormap="Set3", ax=ax)
        ax.set_title(f"{col1} vs {col2}")
        ax.set_ylabel("Proporción")
        plt.xticks(rotation=45)
        return fig

# -------------------------------------------------------------------------------------------------------
# --- Configuración inicial
# -------------------------------------------------------------------------------------------------------
st.set_page_config(layout="centered")  # Opcional: reduce renderizado


# -------------------------------------------------------------------------------------------------------
# --- Funciones cache
# -------------------------------------------------------------------------------------------------------
@st.cache_data
def cargar_dataset(uploaded_file):

    if uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        st.info("El archivo CSV fue convertido a Parquet en memoria.")
    return df


# -------------------------------------------------------------------------------------------------------
# --- Menu lateral / Sidebar
# -------------------------------------------------------------------------------------------------------
st.sidebar.title("Menú de navegación")
menu = st.sidebar.radio("Selecciona un módulo:", 
    ["Módulo 1", 
     "Módulo 2", 
     "Módulo 3"])

# -------------------------------------------------------------------------------------------------------
# --- Módulo 1: Home
# -------------------------------------------------------------------------------------------------------
if menu == "Módulo 1":
    st.title("Telco Customer Churn – Exploratory Data Analysis App")

    st.subheader("Descripción del Proyecto")
    st.write("""Esta aplicación desarrollada con Streamlit realiza un Análisis Exploratorio de Datos (EDA) sobre el dataset TelcoCustomerChurn. El objetivo es identificar patrones y factores asociados a la pérdida de clientes en el sector de telecomunicaciones, aplicando técnicas de limpieza, transformación y visualización de datos. El proyecto integra conceptos de Python, Pandas, NumPy y visualización con Matplotlib y Seaborn, ofreciendo una interfaz intuitiva con sidebar y módulos navegables. Esta herramienta forma parte de mi portafolio profesional como evidencia de competencias en análisis de datos y desarrollo de aplicaciones interactivas.""")

    st.subheader("Datos del Autor:")
    st.write("**Nombre completo:** Carlos Magallanes Loza")
    st.write("**Especialización:** Especialización Python for Analytics")
    st.write("**Año:** 2026")
    
    st.subheader("Explicación del Dataset")
    st.write("El dataset TelcoCustomerChurn reúne información de clientes de una empresa de telecomunicaciones, incluyendo sus características demográficas, servicios contratados y estado de permanencia. El análisis busca explorar qué factores influyen en la decisión de cancelar el servicio.")

    st.subheader("Tecnologías Utilizadas")
    st.markdown("""
    - **Python** 
    - **Pandas**
    - **Numpy**
    - **Matplotlib**                
    - **Seaborn**                
    - **Streamlit**
    - **Visual Studio Code**
    """)


# -------------------------------------------------------------------------------------------------------
# --- Módulo 2: Carga del dataset
# -------------------------------------------------------------------------------------------------------

elif menu == "Módulo 2":
    st.title("Carga del Dataset")

    if 'df' not in st.session_state:
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv","parquet"])

        if uploaded_file is not None:
            try:
                df = cargar_dataset(uploaded_file)
                st.session_state['df'] = df  
                st.success("Archivo cargado correctamente")            

                st.subheader("Vista previa")
                st.dataframe(df.head(200))
                st.write(f"El dataset contiene {df.shape[0]} filas y {df.shape[1]} columnas.")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")
    else:        
        st.success("Dataset ya cargado en memoria, puedes continuar al Módulo 3.")


# -------------------------------------------------------------------------------------------------------
# --- Módulo 3: EDA 1-10 
# -------------------------------------------------------------------------------------------------------

elif menu == "Módulo 3":
    st.title("Análisis Exploratorio de Datos (EDA)")

    # Solo recuperamos el dataset si existe de forma independiente
    if 'df' not in st.session_state:
        st.warning("Primero carga el dataset en el Módulo 2.")
    else:
        df = st.session_state['df']

        analyzer = DataAnalyzer(df)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "1. Información general del dataset", 
            "2. Clasificación de variables", 
            "3. Estadísticas Descriptivas", 
            "4. Valores Faltantes",
            "5. Distribución de variables numéricas",
            "6. Análisis de variables categóricas", 
            "7. Análisis Bivariado (numérico-categórico)",
            "8. Análisis Bivariado (categórico-categórico)", 
            "9. Análisis en Parámetros Selec", 
            "10. Hallazgos clave"
        ])

        # -------------------------------------------------------------------------------------------------------
        with tab1:
            st.subheader("Información general")

            buffer = io.StringIO()          
            df.info(buf=buffer)             
            st.text(buffer.getvalue())      

            st.write("Tipos de datos:")
            st.write(df.dtypes)

            st.write("Conteo de Valores Nulos")
            st.write("Valores nulos:", df.isnull().sum())

        # -------------------------------------------------------------------------------------------------------
        with tab2:
            st.subheader("Clasificación de variables del dataset")

            var_numerica, var_categorica = analyzer.clasificar_variables()

            st.write("Variables numéricas:", list(var_numerica))
            st.write(f"Total numéricas: {len(var_numerica)}")

            st.write("Variables categóricas:", list(var_categorica))
            st.write(f"Total categóricas: {len(var_categorica)}")

            resumen = {"Numéricas": len(var_numerica), "Categóricas": len(var_categorica)}
            st.bar_chart(pd.Series(resumen))

        # -------------------------------------------------------------------------------------------------------        
        with tab3:
            st.subheader("Estadísticas descriptivas")
            st.write(df.describe())

            num_col = st.selectbox(
                "Selecciona una variable numérica para ver métricas:",
                analyzer.var_numerica,
                key="num_item3"
            )

            stats = analyzer.estadisticas_descriptivas(num_col)

            col1, col2, col3 = st.columns(3)
            col1.metric("Media", stats["media"])
            col2.metric("Mediana", stats["mediana"])
            col3.metric("Desviación", stats["desviacion"])


            st.markdown("**Interpretación:**")
            st.write("- La **media** indica el valor promedio de cada variable numérica.")
            st.write("- La **mediana (50%)** refleja el valor central, útil cuando hay sesgos.")
            st.write("- El **std (desviación estándar)** muestra la dispersión respecto a la media.")
            st.write("- Los valores mínimos y máximos ayudan a detectar posibles outliers.")

        # -------------------------------------------------------------------------------------------------------
        with tab4:
            st.subheader("Análisis de Valores Faltantes")

            missing = analyzer.valores_faltantes()
            st.write("Conteo de valores faltantes por columna:")
            st.write(missing)
            st.bar_chart(missing)

            st.markdown("**Discusión:**")
            st.write("Si alguna columna presenta valores nulos, se sugiere realizar lo siguiente:")
            st.markdown("""
            - Eliminar filas o columnas con muchos nulos.
            - Imputar valores (media, mediana, moda o técnicas más avanzadas).
            """)

        # -------------------------------------------------------------------------------------------------------
        with tab5:
            st.subheader("Distribución de variables numéricas")
            
            num_columna1 = st.selectbox(
                "Selecciona una variable numérica:", 
                analyzer.var_numerica, 
                key="varnum5")

            bins = st.slider(
                "Número de bins (intervalos del histograma):", 
                min_value=5, max_value=50, value=20, 
                key="bins_item5"
            )

            mostrar_kde = st.checkbox("Mostrar curva KDE", value=True, key="kde_item5")

            fig = analyzer.plot_histograma(num_columna1, bins=bins, kde=mostrar_kde)
            st.pyplot(fig)
            plt.close(fig)

            stats = analyzer.estadisticas_descriptivas(num_columna1)
            col1, col2, col3 = st.columns(3)
            col1.metric("Media", stats["media"])
            col2.metric("Mediana", stats["mediana"])
            col3.metric("Desviación", stats["desviacion"])

            st.markdown("**Interpretación visual:**")
            st.write("- El histograma muestra la frecuencia de los valores de la variable seleccionada.")
            st.write("- La curva KDE ayuda a identificar la forma de la distribución.")
            st.write("- Si la distribución es simétrica, la media y la mediana estarán cercanas.")
            st.write("- Si está sesgada, puede indicar concentración de clientes en ciertos rangos.")
            st.write("- Valores extremos (outliers) se observan como barras aisladas.")      


        # -------------------------------------------------------------------------------------------------------
        with tab6:
            st.subheader("Análisis de variables categoricas")

            var_categorica1 = df.select_dtypes(include=['object']).columns
            cat_columna1 = st.selectbox("Selecciona una variable categórica:", var_categorica1, key="catnum6")

            conteo = df[cat_columna1].value_counts()        
            st.markdown("**Conteo de categorías:**")
            st.write(conteo)            

            fig, ax = plt.subplots()
            sns.countplot(x=df[cat_columna1], ax=ax, palette="pastel")
            plt.xticks(rotation=45)
            ax.set_title(f"Distribución de {cat_columna1}")
            st.pyplot(fig)
            plt.close(fig)

            proporciones = df[cat_columna1].value_counts(normalize=True)
            st.markdown("**Proporciones (%):**")
            if st.checkbox("Mostrar proporciones", key="prop_item6"):
                st.write((proporciones * 100).round(2))


        # -------------------------------------------------------------------------------------------------------
        with tab7:
            st.subheader("Análisis Bivariado (numérico vs categórico)")

            num_col = st.selectbox("Selecciona una variable numérica:", analyzer.var_numerica, key="varnum7")
            cat_col = st.selectbox("Selecciona una variable categórica:", analyzer.var_categorica, key="catnum7")

            fig = analyzer.plot_boxplot(num_col, cat_col)
            st.pyplot(fig)
            plt.close(fig)


        # -------------------------------------------------------------------------------------------------------
        with tab8: 
            st.subheader("Análisis Bivariado (categórico vs categórico)")

            col1 = st.selectbox("Selecciona la primera variable categórica:", analyzer.var_categorica, key="catnum8_a")
            col2 = st.selectbox("Selecciona la segunda variable categórica:", analyzer.var_categorica, key="catnum8_b")

            tabla = analyzer.tabla_contingencia(col1, col2)
            st.markdown("**Tabla de contingencia (primeras 20 filas):**")
            st.write(tabla)

            fig = analyzer.plot_contingencia(col1, col2)
            st.pyplot(fig)
            plt.close(fig)


        # -------------------------------------------------------------------------------------------------------
        with tab9: 
            st.subheader("Análisis dinámico según parámetros seleccionados")

            tipo_analisis = st.selectbox(
                "Selecciona el tipo de análisis:",
                ["Histograma", "Boxplot", "Gráfico de barras"],
                key="tipo_item9"
            )

            columnas_seleccionadas = st.multiselect(
                "Selecciona columnas para analizar:",
                df.columns,
                key="cols_item9"
            )

            if tipo_analisis == "Histograma":
                for col in columnas_seleccionadas:
                    if col in analyzer.var_numerica:
                        fig = analyzer.plot_histograma(col, bins=20, kde=True)
                        st.pyplot(fig)
                        plt.close(fig)

            elif tipo_analisis == "Boxplot":
                if len(columnas_seleccionadas) == 2:
                    num_col = [c for c in columnas_seleccionadas if c in analyzer.var_numerica]
                    cat_col = [c for c in columnas_seleccionadas if c in analyzer.var_categorica]
                    if num_col and cat_col:
                        fig = analyzer.plot_boxplot(num_col[0], cat_col[0])
                        st.pyplot(fig)
                        plt.close(fig)

            elif tipo_analisis == "Gráfico de barras":
                for col in columnas_seleccionadas:
                    if col in analyzer.var_categorica:
                        fig = analyzer.plot_barras(col)
                        st.pyplot(fig)
                        plt.close(fig)

            st.markdown("**Interpretación dinámica:**")
            st.write("- El usuario controla qué columnas analizar y qué tipo de gráfico mostrar.")
            st.write("- Esto permite explorar relaciones específicas sin sobrecargar la aplicación.")
            st.write("- Los gráficos se adaptan automáticamente según el tipo de variable seleccionada.")


        # -------------------------------------------------------------------------------------------------------
        with tab10:
            st.subheader("Hallazgos clave")

            if "Churn" in df.columns:
                st.markdown("**Resumen visual de Churn:**")
                conteo_churn = df["Churn"].value_counts(normalize=True) * 100
                st.bar_chart(conteo_churn)

            st.markdown("**Interpretación:**")
            st.write("- La visualización muestra la proporción de clientes que cancelan vs. permanecen.")
            st.markdown("**Insights principales:**")
            st.write("- La mayoría de clientes permanecen, pero un porcentaje significativo cancela.")
            st.write("- Variables como `tenure` (tiempo de permanencia) y `PhoneService` muestran patrones claros.")
            st.write("- Los clientes con contratos más cortos tienden a cancelar más.")
            st.write("- Servicios adicionales (Internet, seguridad online) parecen influir en la retención.")



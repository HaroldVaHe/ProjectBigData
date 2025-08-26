# 📊 Crime Data Analysis Project

Este proyecto tiene como objetivo analizar y transformar un dataset de **Crime Data** con más de **1 millón de registros**, explorando distintos formatos de almacenamiento y comparando eficiencia en tamaño y consistencia de datos.  

---

## 📌 Objetivos del Proyecto
1. Seleccionar una base de datos pública de **delitos** (Crime Data).
2. Procesar y explorar los datos en **JSON**, **CSV** y **Parquet**.
3. Validar que los datos sean consistentes entre formatos.
4. Comparar **eficiencia en almacenamiento** (tamaño en disco).
5. Documentar ventajas y desventajas de cada formato.
6. Dejar un entorno reproducible para futuros análisis.

---

## 📂 Estructura del Proyecto
📦 crime-data-project
 ┣ 📂 data
 ┃ ┣ Crime_Data_from_2020_to_Present.parquet
 ┣ 📂 scripts
 ┃ ┣ ExtraccionDeDatos.py
 ┗ README.md

## 🛠️ Tecnologías utilizadas

Python 3.x

Pandas → Transformación y validación de datos.

PyArrow / Fastparquet → Soporte para formato Parquet.

OS / Sys → Manejo de tamaños de archivos.

## 🚀 Flujo del Proyecto
1️⃣ Selección de la Base de Datos

Se seleccionó un dataset público de Crime Data con 1,004,991 registros, el cual contiene información de delitos, ubicaciones, fechas y otros metadatos.

2️⃣ Conversión de Formatos

JSON → Parquet usando pandas y pyarrow.

JSON → CSV para comparar con un formato ampliamente usado.

3️⃣ Validación de Consistencia

El script ExtraccionDeDatos.py:

Convierte los datos de JSON a Parquet.

Valida que el número de registros y columnas sea igual.

Comprueba diferencias de contenido.

4️⃣ Comparación de Tamaños

Calcula el tamaño en disco de JSON, CSV y Parquet.

Reporta la diferencia y reducción de almacenamiento.

## 📊 Conclusiones

Parquet es el formato más eficiente en almacenamiento y velocidad de análisis.

CSV es útil para compatibilidad, pero requiere cuidado con formatos de fecha y tipos de datos.

JSON es recomendable para transmisión en tiempo real o APIs, pero no como formato principal de análisis.


## 📈 Futuras Extensiones

- Integración con **Apache Spark** para procesamiento distribuido.  
- Uso de **Data Streaming (Kafka o APIs)** para ingestión en tiempo real sin depender de archivos intermedios.  
- Dashboard en **Power BI / Tableau** para visualización de estadísticas del crimen.  

---

## 👨‍💻 Autor

Proyecto desarrollado por **David Lopez**, **Martin Jerez**, **Juliana Espinel** y **Harold Vargas** 

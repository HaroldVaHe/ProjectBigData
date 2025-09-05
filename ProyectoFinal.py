#!/usr/bin/env python
# coding: utf-8

# # PROYECTO BIG  DATA 

# Integrantes: Martín Jerez, Harold Vargas, Juliana Espinel y David Lopez 

# ### Pregunta de investigación

# ¿Cómo pueden detectarse patrones espacio-temporales de criminalidad en la ciudad de Los Ángeles mediante el análisis de grandes volúmenes de reportes, con el fin de anticipar zonas de riesgo y optimizar la toma de decisiones en seguridad pública?

# ### Objetivo

# Analizar grandes volúmenes de reportes de criminalidad en la ciudad de Los Ángeles para detectar patrones espacio-temporales que permitan anticipar zonas de riesgo y fortalecer la toma de decisiones en materia de seguridad pública.

# ### Glosario

# # Diccionario de Datos - Crime Dataset (LAPD)
# 
# | **Campo**       | **Descripción** | **Nombre en minúscula** | **Tipo de dato** |
# |-----------------|-----------------|-------------------------|------------------|
# | DR_NO           | Division of Records Number: Official file number made up of a 2 digit year, area ID, and 5 digits | `dr_no` | Text |
# | Date Rptd       | Fecha en formato MM/DD/YYYY | `date_rptd` | Floating Timestamp |
# | DATE OCC        | Fecha de ocurrencia en formato MM/DD/YYYY | `date_occ` | Floating Timestamp |
# | TIME OCC        | En formato militar (24 horas) | `time_occ` | Text |
# | AREA            | Número de área (1–21) correspondiente a una estación de policía LAPD | `area` | Text |
# | AREA NAME       | Nombre de la estación/división de policía correspondiente al área | `area_name` | Text |
# | Rpt Dist No     | Código de 4 dígitos que representa un sub-área dentro de un Área Geográfica | `rpt_dist_no` | Text |
# | Part 1-2        | Clasificación del crimen (Parte 1 o 2) | `part_1_2` | Number |
# | Crm Cd          | Código del crimen cometido (igual a Crime Code 1) | `crm_cd` | Text |
# | Crm Cd Desc     | Descripción del código de crimen | `crm_cd_desc` | Text |
# | Mocodes         | Modus Operandi: Actividades asociadas con el sospechoso [(lista de códigos MO)](https://data.lacity.org/api/views/y8tr-7khq/files/3a967fbd-f210-4857-bc52-60230efe256c?download=true&filename=MO%20CODES%20(numerical%20order).pdf) | `mocodes` | Text |
# | Vict Age        | Edad de la víctima (número de 2 dígitos) | `vict_age` | Text |
# | Vict Sex        | Sexo de la víctima: F = Femenino, M = Masculino, X = Desconocido | `vict_sex` | Text |
# | Vict Descent    | Código de ascendencia/raza de la víctima (ej: B = Black, H = Hispanic, W = White, etc.) | `vict_descent` | Text |
# | Premis Cd       | Código del tipo de estructura, vehículo o lugar donde ocurrió el crimen | `premis_cd` | Number |
# | Premis Desc     | Descripción del código de premisa | `premis_desc` | Text |
# | Weapon Used Cd  | Código del arma utilizada en el crimen | `weapon_used_cd` | Text |
# | Weapon Desc     | Descripción del arma utilizada | `weapon_desc` | Text |
# | Status          | Estado del caso (ej: IC = default) | `status` | Text |
# | Status Desc     | Descripción del estado | `status_desc` | Text |
# | Crm Cd 1        | Código del crimen principal (más serio) | `crm_cd_1` | Text |
# | Crm Cd 2        | Código de crimen adicional (menos serio que Crm Cd 1) | `crm_cd_2` | Text |
# | Crm Cd 3        | Código de crimen adicional (menos serio que Crm Cd 2) | `crm_cd_3` | Text |
# | Crm Cd 4        | Código de crimen adicional (menos serio que Crm Cd 3) | `crm_cd_4` | Text |
# | LOCATION        | Dirección aproximada del incidente (redondeada a la centena para anonimato) | `location` | Text |
# | Cross Street    | Calle transversal de la dirección aproximada | `cross_street` | Text |
# | LAT             | Latitud | `lat` | Number |
# | LON             | Longitud | `lon` | Number |
# 

# ## Conectar con mi contenedor 

# """
# Script completo para cargar datos de criminalidad LAPD desde Parquet a PostgreSQL
# Incluye creación automática de tabla y carga de datos
# Asegúrate de instalar las dependencias:
# pip install pandas pyarrow psycopg2-binary sqlalchemy
# """
import time
from scipy import stats
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import folium
import statsmodels.api as sm
from statsmodels.formula.api import ols
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
import logging
import traceback
warnings.filterwarnings('ignore')

# Configuración de la conexión a PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': '5433',
    'database': 'BigdataFinal',
    'user': 'psqluser',
    'password': 'psqlpass'
}

# Ruta del archivo (cambia esta ruta por la tuya)
BASE_DIR = os.path.abspath(os.path.dirname(''))  # carpeta donde está el script
PARQUET_FILE_PATH = os.path.join(BASE_DIR, "Crime_Data_from_2020_to_Present.parquet")
# Timing decorator to report task duration
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Task '{func.__name__}' took {duration:.4f} seconds")
        return result
    return wrapper
@timing_decorator
@task(name="Crear conexión a PostgreSQL", cache_policy=NO_CACHE)
def create_connection():
    # """Crear conexión a PostgreSQL"""
    logger = get_run_logger()
    try:
        # Para psycopg2
        conn = psycopg2.connect(**DB_CONFIG)

        # Para SQLAlchemy
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        logger.info("✅ Conexión exitosa a PostgreSQL")
        return conn, engine

    except Exception as e:
        logger.info(f"❌ Error conectando a PostgreSQL: {e}")
        return None, None

@timing_decorator
@task(name="Verificar si la tabla tiene datos", cache_policy=NO_CACHE)
def table_has_data(conn):
    # """Verificar si la tabla lapd_crime_data tiene datos"""
    logger = get_run_logger()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM lapd_crime_data;")
            count = cursor.fetchone()[0]
            logger.info(f"✅ La tabla tiene {count} registros.")
            return count > 0
    except Exception as e:
        # Si hay algún error (como que la tabla no existe), asumimos que no hay datos
        logger.info(f"⚠️  Error al verificar datos: {e}. Asumiendo que no hay datos.")
        return False

@timing_decorator
@task(name="Crear tabla lapd_crime_data", cache_policy=NO_CACHE)
def create_table(conn):
    # """Crear la tabla lapd_crime_data si no existe"""
    logger = get_run_logger()
    logger.info("🏗️  Verificando si existe la tabla lapd_crime_data...")

    # Verificar si la tabla ya existe
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'lapd_crime_data'
            );
        """)
        table_exists = cursor.fetchone()[0]

    if table_exists:
        logger.info("✅ La tabla ya existe.")
        return True
    else:
        logger.info("📋 La tabla no existe. Creándola...")
        create_table_sql = """
        CREATE TABLE lapd_crime_data (
            dr_no VARCHAR(20) PRIMARY KEY,
            date_rptd DATE,
            date_occ DATE,
            time_occ TIME,
            area VARCHAR(10),
            area_name VARCHAR(100),
            rpt_dist_no VARCHAR(10),
            part_1_2 INTEGER,
            crm_cd VARCHAR(10),
            crm_cd_desc VARCHAR(200),
            mocodes VARCHAR(100),
            vict_age VARCHAR(10),
            vict_sex CHAR(1),
            vict_descent CHAR(1),
            premis_cd INTEGER,
            premis_desc VARCHAR(200),
            weapon_used_cd VARCHAR(10),
            weapon_desc VARCHAR(200),
            status VARCHAR(10),
            status_desc VARCHAR(100),
            crm_cd_1 VARCHAR(10),
            crm_cd_2 VARCHAR(10),
            crm_cd_3 VARCHAR(10),
            crm_cd_4 VARCHAR(10),
            location VARCHAR(200),
            cross_street VARCHAR(200),
            lat DECIMAL(10, 8),
            lon DECIMAL(11, 8)
        );
        """
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
            conn.commit()
            logger.info("✅ Tabla creada exitosamente")

            # Ahora crear los índices
            create_index_sql = """
            CREATE INDEX idx_lapd_date_occ ON lapd_crime_data(date_occ);
            CREATE INDEX idx_lapd_area ON lapd_crime_data(area);
            CREATE INDEX idx_lapd_crm_cd ON lapd_crime_data(crm_cd);
            CREATE INDEX idx_lapd_location ON lapd_crime_data(lat, lon);
            CREATE INDEX idx_lapd_area_date ON lapd_crime_data(area, date_occ);
            """
            with conn.cursor() as cursor:
                cursor.execute(create_index_sql)
            conn.commit()
            logger.info("✅ Índices creados exitosamente")
            return True
        except Exception as e:
            logger.info(f"❌ Error creando tabla o índices: {e}")
            conn.rollback()
            return False

@timing_decorator
@task(name="Limpiar y preparar datos", cache_policy=NO_CACHE)
def clean_data(df):
    # """Limpiar y preparar los datos"""
    logger = get_run_logger()
    logger.info("🧹 Limpiando datos...")

    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    logger.info(df_clean.head())
    logger.info(df_clean.shape)

    # Mostrar las columnas originales
    logger.info("📋 Columnas encontradas en el archivo:")
    for i, col in enumerate(df_clean.columns):
        logger.info(f"  {i+1}. {col}")

    # Mapear nombres de columnas del archivo original a nombres de BD
    column_mapping = {
        'DR_NO': 'dr_no',
        'Date Rptd': 'date_rptd',
        'DATE OCC': 'date_occ', 
        'TIME OCC': 'time_occ',
        'AREA': 'area',
        'AREA NAME': 'area_name',
        'Rpt Dist No': 'rpt_dist_no',
        'Part 1-2': 'part_1_2',
        'Crm Cd': 'crm_cd',
        'Crm Cd Desc': 'crm_cd_desc',
        'Mocodes': 'mocodes',
        'Vict Age': 'vict_age',
        'Vict Sex': 'vict_sex',
        'Vict Descent': 'vict_descent',
        'Premis Cd': 'premis_cd',
        'Premis Desc': 'premis_desc',
        'Weapon Used Cd': 'weapon_used_cd',
        'Weapon Desc': 'weapon_desc',
        'Status': 'status',
        'Status Desc': 'status_desc',
        'Crm Cd 1': 'crm_cd_1',
        'Crm Cd 2': 'crm_cd_2',
        'Crm Cd 3': 'crm_cd_3',
        'Crm Cd 4': 'crm_cd_4',
        'LOCATION': 'location',
        'Cross Street': 'cross_street',
        'LAT': 'lat',
        'LON': 'lon'
    }

    # Renombrar columnas
    df_clean = df_clean.rename(columns=column_mapping)

    # Limpiar coordenadas inválidas (0, 0)
    if 'lat' in df_clean.columns and 'lon' in df_clean.columns:
        df_clean.loc[(df_clean['lat'] == 0) | (df_clean['lon'] == 0), ['lat', 'lon']] = np.nan

    # Convertir fechas
    date_columns = ['date_rptd', 'date_occ']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce').dt.date

    # Limpiar tiempo
    if 'time_occ' in df_clean.columns:
        def format_time(time_val):
            if pd.isna(time_val):
                return None
            time_str = str(int(time_val)).zfill(4) if isinstance(time_val, (int, float)) else str(time_val).zfill(4)
            if len(time_str) >= 4:
                return f"{time_str[:2]}:{time_str[2:4]}"
            return None

        df_clean['time_occ'] = df_clean['time_occ'].apply(format_time)

    # Limpiar edad de víctima
    if 'vict_age' in df_clean.columns:
        df_clean['vict_age'] = df_clean['vict_age'].replace([0, -1], np.nan)
        df_clean['vict_age'] = df_clean['vict_age'].astype(str)

    # Truncar strings largos para evitar errores
    string_columns = {
        'area_name': 100,
        'crm_cd_desc': 200,
        'mocodes': 100,
        'premis_desc': 200,
        'weapon_desc': 200,
        'status_desc': 100,
        'location': 200,
        'cross_street': 200
    }

    for col, max_len in string_columns.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str[:max_len]

    # Asegurar que dr_no no tenga valores nulos
    if 'dr_no' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['dr_no'])
        df_clean['dr_no'] = df_clean['dr_no'].astype(str)

    logger.info(f"✅ Datos limpiados. Registros: {len(df_clean)}")
    logger.info(df_clean.head())
    logger.info(df_clean.shape)

    return df_clean

@timing_decorator
@task(name="Cargar archivo Parquet a PostgreSQL", cache_policy=NO_CACHE)
def load_parquet_to_postgres(batch_size=5000):
    # """Cargar archivo Parquet a PostgreSQL"""

    logger = get_run_logger()

    # Verificar que el archivo existe
    if not os.path.exists(PARQUET_FILE_PATH):
        logger.info(f"❌ Archivo no encontrado: {PARQUET_FILE_PATH}")
        return False

    # Crear conexión
    conn, engine = create_connection()
    if not conn or not engine:
        return False

    try:
        # Crear tabla si no existe
        if not create_table(conn):
            logger.info("❌ Falló la creación de la tabla. Abortando.")
            return False

        # Verificar si la tabla ya tiene datos
        if table_has_data(conn):
            logger.info("✅ La tabla ya tiene datos. No se cargarán datos duplicados.")
            return True

        logger.info(f"📁 Cargando archivo: {PARQUET_FILE_PATH}")

        # Leer el archivo Parquet
        logger.info("📖 Leyendo archivo Parquet...")
        df = pd.read_parquet(PARQUET_FILE_PATH)

        logger.info(f"📊 Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")
        logger.info(f"📏 Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Limpiar datos
        df_clean = clean_data(df)

        # Cargar datos en lotes
        logger.info(f"💾 Cargando datos en lotes de {batch_size:,} registros...")

        total_batches = len(df_clean) // batch_size + (1 if len(df_clean) % batch_size > 0 else 0)
        loaded_records = 0

        for i, start_idx in enumerate(range(0, len(df_clean), batch_size)):
            end_idx = min(start_idx + batch_size, len(df_clean))
            batch_df = df_clean.iloc[start_idx:end_idx]

            try:
                # Cargar lote
                batch_df.to_sql(
                    'lapd_crime_data',
                    engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

                loaded_records += len(batch_df)
                progress = ((i + 1) / total_batches) * 100
                logger.info(f"📈 Progreso: {progress:.1f}% - Lote {i+1}/{total_batches} - Registros cargados: {loaded_records:,}")

            except Exception as e:
                logger.info(f"⚠️  Error en lote {i+1}: {e}")
                continue

        logger.info("✅ Datos cargados exitosamente!")

        # Verificar carga
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM lapd_crime_data;")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Total de registros en la base de datos: {count:,}")

            # Mostrar algunas estadísticas
            cursor.execute("""
                SELECT 
                    MIN(date_occ) as fecha_min,
                    MAX(date_occ) as fecha_max,
                    COUNT(DISTINCT area) as areas_distintas,
                    COUNT(DISTINCT crm_cd) as tipos_crimen
                FROM lapd_crime_data;
            """)
            stats = cursor.fetchone()
            logger.info(f"📈 Estadísticas:")
            logger.info(f"  - Rango de fechas: {stats[0]} a {stats[1]}")
            logger.info(f"  - Áreas distintas: {stats[2]}")
            logger.info(f"  - Tipos de crimen distintos: {stats[3]}")

        return True

    except Exception as e:
        logger.info(f"❌ Error cargando datos: {e}")
        import traceback
        traceback.logger.info_exc()
        return False

    finally:
        if conn:
            conn.close()
        if engine:
            engine.dispose()





# ## Missing values 


@timing_decorator
@task(name="Analizar missing values en lapd_crime_data", cache_policy=NO_CACHE)
def analyze_missing_values(GraficPath='missing_values_analysis.png'):
    # """Analizar missing values en la tabla lapd_crime_data"""
    logger = get_run_logger()
    conn, engine = create_connection()  # Desempaquetar la tupla
    if conn is None or engine is None:
        return

    try:
        # Obtener el total de registros
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM lapd_crime_data;")
            total_records = cursor.fetchone()[0]
            logger.info(f"📊 Total de registros en la tabla: {total_records:,}")

        # Obtener la lista de columnas
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'lapd_crime_data'
                ORDER BY ordinal_position;
            """)
            columns = [row[0] for row in cursor.fetchall()]

        logger.info("\n🔍 Analizando missing values...")

        # Analizar cada columna
        results = []
        for column in columns:
            with conn.cursor() as cursor:
                # Contar valores NULL
                cursor.execute(f"SELECT COUNT(*) FROM lapd_crime_data WHERE {column} IS NULL;")
                null_count = cursor.fetchone()[0]

                # Para columnas específicas, verificar valores que representan missing values
                special_missing = 0
                if column in ['lat', 'lon']:
                    cursor.execute(f"SELECT COUNT(*) FROM lapd_crime_data WHERE {column} = 0;")
                    special_missing = cursor.fetchone()[0]
                elif column == 'vict_age':
                    cursor.execute(f"SELECT COUNT(*) FROM lapd_crime_data WHERE {column} IN ('0', '-1', 'NaN');")
                    special_missing = cursor.fetchone()[0]

                total_missing = null_count + special_missing
                missing_percentage = (total_missing / total_records) * 100

                results.append({
                    'column': column,
                    'null_count': null_count,
                    'special_missing': special_missing,
                    'total_missing': total_missing,
                    'missing_percentage': missing_percentage
                })

        # Crear DataFrame con los resultados
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('missing_percentage', ascending=False)

        # Mostrar resultados
        logger.info("\n" + "="*80)
        logger.info("ANÁLISIS DE VALORES FALTANTES")
        logger.info("="*80)
        for _, row in df_results.iterrows():
            logger.info(f"{row['column']:20} | {row['total_missing']:>8,} | {row['missing_percentage']:>6.2f}% | "
                  f"(NULL: {row['null_count']:,}, Especial: {row['special_missing']:,})")

        # Visualizar resultados
        plt.figure(figsize=(12, 8))
        bars = plt.barh(df_results['column'], df_results['missing_percentage'], color='skyblue')
        plt.xlabel('Porcentaje de valores faltantes')
        plt.title('Porcentaje de valores faltantes por columna')
        plt.gca().invert_yaxis()  # Mostrar la columna con mayor porcentaje en la parte superior

        # Añadir etiquetas de valor en las barras
        for bar, percentage in zip(bars, df_results['missing_percentage']):
            if percentage > 0:
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{percentage:.1f}%', ha='left', va='center')

        plt.tight_layout()
        plt.savefig(GraficPath, dpi=300, bbox_inches='tight')
        logger.info("\n📈 Gráfico guardado como 'missing_values_analysis.png'")

        # Mostrar estadísticas resumidas
        logger.info("\n" + "="*80)
        logger.info("ESTADÍSTICAS RESUMEN")
        logger.info("="*80)
        logger.info(f"Total de registros: {total_records:,}")
        logger.info(f"Columnas con más del 50% de valores faltantes: {len(df_results[df_results['missing_percentage'] > 50])}")
        logger.info(f"Columnas con menos del 5% de valores faltantes: {len(df_results[df_results['missing_percentage'] < 5])}")

        # Mostrar las 5 columnas con más valores faltantes
        logger.info("\nTop 5 columnas con más valores faltantes:")
        top5 = df_results.head()
        for _, row in top5.iterrows():
            logger.info(f"  - {row['column']}: {row['missing_percentage']:.2f}%")

        # Guardar resultados en CSV
        df_results.to_csv(GraficPath, index=False)
        logger.info(f"\n📊 Resultados detallados guardados en '{GraficPath}'")

    except Exception as e:
        logger.info(f"❌ Error analizando missing values: {e}")
        import traceback
        traceback.logger.info_exc()

    finally:
        if conn:
            conn.close()  # Cerrar solo la conexión psycopg2
        if engine:
            engine.dispose()  # Cerrar el motor de SQLAlchemy

@timing_decorator
@task(name="Limpiar datos en lapd_crime_data", cache_policy=NO_CACHE)
def drop_columns(conn):
    # """Eliminar columnas con muchos valores nulos"""
    logger = get_run_logger()
    logger.info("🗑️  Eliminando columnas con muchos nulos...")
    try:
        with conn.cursor() as cursor:
            # Verificar si las columnas existen antes de eliminarlas
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'lapd_crime_data' 
                AND column_name IN ('crm_cd_2', 'crm_cd_3', 'crm_cd_4', 'weapon_used_cd', 'weapon_desc');
            """)
            existing_columns = [row[0] for row in cursor.fetchall()]

            if existing_columns:
                drop_columns_sql = "ALTER TABLE lapd_crime_data "
                drop_columns_sql += ", ".join([f"DROP COLUMN {col}" for col in existing_columns])
                cursor.execute(drop_columns_sql)
                logger.info(f"✅ Columnas eliminadas: {existing_columns}")
            else:
                logger.info("✅ No hay columnas para eliminar")
        conn.commit()
    except Exception as e:
        logger.info(f"❌ Error eliminando columnas: {e}")
        conn.rollback()

@timing_decorator
@task(name="Limpiar valores especiales en lapd_crime_data", cache_policy=NO_CACHE)
def clean_special_values(conn):
    # """Limpiar valores especiales (0 en coordenadas, valores inválidos en vict_age)"""
    logger = get_run_logger()
    logger.info("🧹 Limpiando valores especiales...")
    try:
        with conn.cursor() as cursor:
            # 1. Para lat y lon: convertir 0 a NULL
            cursor.execute("""
                UPDATE lapd_crime_data 
                SET lat = NULL 
                WHERE lat = 0;
            """)
            lat_updated = cursor.rowcount
            cursor.execute("""
                UPDATE lapd_crime_data 
                SET lon = NULL 
                WHERE lon = 0;
            """)
            lon_updated = cursor.rowcount
            logger.info(f"✅ Coordenadas 0 convertidas a NULL: lat={lat_updated}, lon={lon_updated}")

            # 2. Para vict_age: convertir valores no numéricos y edades inválidas a NULL
            cursor.execute("""
                UPDATE lapd_crime_data 
                SET vict_age = NULL 
                WHERE vict_age IS NOT NULL AND (
                    vict_age !~ '^[0-9]+$' OR 
                    vict_age::integer <= 0 OR 
                    vict_age::integer > 120
                );
            """)
            vict_age_updated = cursor.rowcount
            logger.info(f"✅ Valores inválidos en vict_age convertidos a NULL: {vict_age_updated}")

        conn.commit()
    except Exception as e:
        logger.info(f"❌ Error limpiando valores especiales: {e}")
        conn.rollback()

@timing_decorator
@task(name="Eliminar filas con nulos en columnas clave", cache_policy=NO_CACHE)
def delete_rows_with_nulls(conn):
    # """Eliminar filas con valores nulos en columnas clave"""
    logger = get_run_logger()
    logger.info("🧹 Eliminando filas con nulos en columnas clave...")
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM lapd_crime_data 
                WHERE lon IS NULL 
                OR lat IS NULL 
                OR premis_cd IS NULL 
                OR crm_cd_1 IS NULL 
                OR status IS NULL;
            """)
            deleted_rows = cursor.rowcount
            conn.commit()
            logger.info(f"✅ Filas eliminadas: {deleted_rows}")
    except Exception as e:
        logger.info(f"❌ Error eliminando filas: {e}")
        conn.rollback()

@timing_decorator
@task(name="Imputar valores missing en lapd_crime_data", cache_policy=NO_CACHE)
def impute_vict_sex(conn):
    # """Imputar valores missing para vict_sex con la moda"""
    logger = get_run_logger()
    logger.info("🔧 Imputando vict_sex...")
    try:
        with conn.cursor() as cursor:
            # Obtener la moda (valor más frecuente)
            cursor.execute("""
                SELECT vict_sex, COUNT(*) as count 
                FROM lapd_crime_data 
                WHERE vict_sex IS NOT NULL AND vict_sex != 'nan'
                GROUP BY vict_sex 
                ORDER BY count DESC 
                LIMIT 1;
            """)
            mode = cursor.fetchone()
            if mode:
                mode_value = mode[0]
                logger.info(f"Moda para vict_sex: {mode_value}")
                # Actualizar los valores missing
                cursor.execute("""
                    UPDATE lapd_crime_data 
                    SET vict_sex = %s
                    WHERE vict_sex IS NULL OR vict_sex = 'nan';
                """, (mode_value,))
                updated_rows = cursor.rowcount
                conn.commit()
                logger.info(f"✅ vict_sex imputado. Filas actualizadas: {updated_rows}")
            else:
                logger.info("⚠️  No se encontró moda para vict_sex")
    except Exception as e:
        logger.info(f"❌ Error imputando vict_sex: {e}")
        conn.rollback()

@timing_decorator
@task(name="Imputar valores missing en lapd_crime_data", cache_policy=NO_CACHE)
def impute_vict_descent(conn):
    # """Imputar valores missing para vict_descent con la moda"""
    logger = get_run_logger()
    logger.info("🔧 Imputando vict_descent...")
    try:
        with conn.cursor() as cursor:
            # Obtener la moda
            cursor.execute("""
                SELECT vict_descent, COUNT(*) as count 
                FROM lapd_crime_data 
                WHERE vict_descent IS NOT NULL AND vict_descent != 'nan'
                GROUP BY vict_descent 
                ORDER BY count DESC 
                LIMIT 1;
            """)
            mode = cursor.fetchone()
            if mode:
                mode_value = mode[0]
                logger.info(f"Moda para vict_descent: {mode_value}")
                # Actualizar los valores missing
                cursor.execute("""
                    UPDATE lapd_crime_data 
                    SET vict_descent = %s
                    WHERE vict_descent IS NULL OR vict_descent = 'nan';
                """, (mode_value,))
                updated_rows = cursor.rowcount
                conn.commit()
                logger.info(f"✅ vict_descent imputado. Filas actualizadas: {updated_rows}")
            else:
                logger.info("⚠️  No se encontró moda para vict_descent")
    except Exception as e:
        logger.info(f"❌ Error imputando vict_descent: {e}")
        conn.rollback()

@timing_decorator
@task(name="Imputar valores missing en lapd_crime_data", cache_policy=NO_CACHE)
def impute_vict_age(conn):
    # """Imputar valores missing para vict_age con la mediana"""
    logger = get_run_logger()
    logger.info("🔧 Imputando vict_age...")
    try:
        with conn.cursor() as cursor:
            # Calcular la mediana
            cursor.execute("""
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vict_age::numeric) 
                FROM lapd_crime_data 
                WHERE vict_age IS NOT NULL;
            """)
            median = cursor.fetchone()[0]
            if median:
                median_value = int(median)  # La mediana puede ser float, pero la edad es entera
                logger.info(f"Mediana para vict_age: {median_value}")
                # Actualizar los valores NULL
                cursor.execute("""
                    UPDATE lapd_crime_data 
                    SET vict_age = %s
                    WHERE vict_age IS NULL;
                """, (str(median_value),))  # Guardar como string
                updated_rows = cursor.rowcount
                conn.commit()
                logger.info(f"✅ vict_age imputado. Filas actualizadas: {updated_rows}")
            else:
                logger.info("⚠️  No se pudo calcular la mediana para vict_age")
    except Exception as e:
        logger.info(f"❌ Error imputando vict_age: {e}")
        conn.rollback()


# ### Valores Atipicos 

@timing_decorator
@task(name="Analizar outliers en lapd_crime_data", cache_policy=NO_CACHE)
def get_numeric_columns(conn):
    # """Obtener las columnas numéricas de la tabla"""
    logger = get_run_logger()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'lapd_crime_data'
                AND data_type IN ('integer', 'numeric', 'double precision', 'real', 'bigint', 'smallint')
                ORDER BY ordinal_position;
            """)
            numeric_columns = [row[0] for row in cursor.fetchall()]
            return numeric_columns
    except Exception as e:
        logger.info(f"❌ Error obteniendo columnas numéricas: {e}")
        return []

@timing_decorator
@task(name="Cargar datos numéricos desde lapd_crime_data", cache_policy=NO_CACHE)
def load_numeric_data(engine, columns):
    # """Cargar datos numéricos desde la base de datos"""
    logger = get_run_logger()
    try:
        # Construir la consulta SQL
        columns_str = ', '.join(columns)
        query = f"SELECT {columns_str} FROM lapd_crime_data;"

        # Cargar los datos en un DataFrame
        df = pd.read_sql_query(query, engine)  # Usar engine en lugar de conn
        return df
    except Exception as e:
        logger.info(f"❌ Error cargando datos: {e}")
        return pd.DataFrame()

@timing_decorator
@task(name="Analizar y visualizar outliers en lapd_crime_data", cache_policy=NO_CACHE)
def analyze_outliers(df):
    # """Analizar y visualizar outliers con boxplots"""
    logger = get_run_logger()
    if df.empty:
        logger.info("❌ No hay datos para analizar")
        return

    logger.info("📊 Analizando outliers...")

    # Configurar el estilo de los gráficos
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Calcular el número de filas y columnas para subplots
    n_cols = 2
    n_rows = (len(df.columns) + n_cols - 1) // n_cols

    # Crear figura para los boxplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Diccionario para almacenar información de outliers
    outliers_info = {}

    for i, column in enumerate(df.columns):
        # Obtener datos sin valores nulos
        data = df[column].dropna()

        if len(data) == 0:
            continue

        # Calcular estadísticas básicas
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificar outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers_percentage = (len(outliers) / len(data)) * 100

        # Almacenar información
        outliers_info[column] = {
            'count': len(outliers),
            'percentage': outliers_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'median': data.median()
        }

        # Crear boxplot
        ax = axes[i]
        sns.boxplot(y=data, ax=ax)
        ax.set_title(f'{column}\nOutliers: {len(outliers)} ({outliers_percentage:.2f}%)')
        ax.set_ylabel('Valor')

        # Añadir líneas para los límites
        ax.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.7, label=f'Límite inferior: {lower_bound:.2f}')
        ax.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.7, label=f'Límite superior: {upper_bound:.2f}')

        if i == 0:  # Solo añadir leyenda al primer gráfico para evitar duplicados
            ax.legend()

    # Ocultar ejes vacíos si los hay
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('boxplots_outliers.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Boxplots guardados como 'boxplots_outliers.png'")

    # Mostrar resumen de outliers
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE OUTLIERS POR VARIABLE")
    logger.info("="*80)

    for column, info in outliers_info.items():
        logger.info(f"\n📈 {column}:")
        logger.info(f"   - Outliers: {info['count']} ({info['percentage']:.2f}%)")
        logger.info(f"   - Rango normal: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        logger.info(f"   - Rango real: [{info['min']:.2f}, {info['max']:.2f}]")
        logger.info(f"   - Media: {info['mean']:.2f}, Mediana: {info['median']:.2f}")

    # Guardar resumen en CSV
    summary_df = pd.DataFrame.from_dict(outliers_info, orient='index')
    summary_df.to_csv('outliers_summary.csv')
    logger.info("\n📊 Resumen de outliers guardado como 'outliers_summary.csv'")

    # Recomendaciones basadas en el análisis
    logger.info("\n" + "="*80)
    logger.info("RECOMENDACIONES")
    logger.info("="*80)

    for column, info in outliers_info.items():
        if info['percentage'] > 5:  # Si más del 5% son outliers
            logger.info(f"⚠️  {column}: Tiene un porcentaje significativo de outliers ({info['percentage']:.2f}%).")
            logger.info(f"   Considera investigar si estos valores son errores o casos legítimos pero extremos.")
            logger.info(f"   Posibles acciones:")
            logger.info(f"   - Aplicar transformación logarítmica")
            logger.info(f"   - Usar winsorization (limitar los valores extremos)")
            logger.info(f"   - Crear variable categórica (ej: grupos de edad para vict_age)")
        else:
            logger.info(f"✅ {column}: Porcentaje de outliers aceptable ({info['percentage']:.2f}%)")




# """
# Script para clustering geográfico con carga por lotes usando Prefect
# """
# Configuración de optimización para grandes volúmenes
BATCH_SIZE = 50000  # Tamaño de lote para carga de datos
MAX_RECORDS = None  # None para todos los registros, o un número para limitar
VISUALIZATION_SAMPLE = 10000  # Tamaño de muestra para visualización

# Coordenadas de Los Ángeles para el mapa
LA_BOUNDS = {
    'min_lat': 33.7,
    'max_lat': 34.4,
    'min_lon': -118.7,
    'max_lon': -118.1
}
@timing_decorator
@task(name="Crear conexión a PostgreSQL", cache_policy=NO_CACHE, retries=3, retry_delay_seconds=10)
def create_connection():
    # """Crear conexión a PostgreSQL"""
    logger = get_run_logger()
    try:
        # Para psycopg2
        conn = psycopg2.connect(**DB_CONFIG)

        # Para SQLAlchemy
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        logger.info("✅ Conexión exitosa a PostgreSQL")
        return conn, engine

    except Exception as e:
        logger.error(f"❌ Error conectando a PostgreSQL: {e}")
        raise

@timing_decorator
@task(name="Estimar total de registros válidos", cache_policy=NO_CACHE)
def estimate_total_records(conn):
    # """Estimar el número total de registros válidos"""
    logger = get_run_logger()
    try:
        with conn.cursor() as cursor:
            query = f"""
            SELECT COUNT(*) 
            FROM lapd_crime_data 
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            AND lat != 0 AND lon != 0
            AND lat BETWEEN {LA_BOUNDS['min_lat']} AND {LA_BOUNDS['max_lat']}
            AND lon BETWEEN {LA_BOUNDS['min_lon']} AND {LA_BOUNDS['max_lon']};
            """
            cursor.execute(query)
            total_records = cursor.fetchone()[0]
            logger.info(f"📊 Total de registros estimados: {total_records:,}")
            return total_records
    except Exception as e:
        logger.error(f"❌ Error estimando total de registros: {e}")
        return 0

@timing_decorator
@task(name="Cargar datos geográficos en lotes", cache_policy=NO_CACHE)
def load_geo_data_in_batches(conn, engine, total_records, batch_size=BATCH_SIZE, max_records=MAX_RECORDS):
    # """Cargar datos geográficos en lotes para manejar grandes volúmenes"""
    logger = get_run_logger()
    try:
        if max_records and max_records < total_records:
            total_records = max_records

        logger.info(f"📊 Total de registros a procesar: {total_records:,}")

        # Calcular número de lotes
        num_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"📦 Procesando en {num_batches} lotes de {batch_size} registros")

        all_data = []
        for batch_num in range(num_batches):
            offset = batch_num * batch_size
            limit = min(batch_size, total_records - offset)

            query = f"""
            SELECT dr_no, lat, lon, date_occ, crm_cd_desc, area_name
            FROM lapd_crime_data 
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            AND lat != 0 AND lon != 0
            AND lat BETWEEN {LA_BOUNDS['min_lat']} AND {LA_BOUNDS['max_lat']}
            AND lon BETWEEN {LA_BOUNDS['min_lon']} AND {LA_BOUNDS['max_lon']}
            ORDER BY dr_no
            LIMIT {limit} OFFSET {offset};
            """

            logger.info(f"📥 Cargando lote {batch_num + 1}/{num_batches}...")
            df_batch = pd.read_sql_query(query, conn)
            all_data.append(df_batch)

            # Liberar memoria
            del df_batch

        # Combinar todos los lotes
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Datos geográficos cargados: {len(df):,} registros")
        return df
    except Exception as e:
        logger.error(f"❌ Error cargando datos geográficos: {e}")
        raise

@timing_decorator
@task(name="Determinar número óptimo de clusters", cache_policy=NO_CACHE)
def determine_optimal_clusters(data, max_clusters=8):
    # """Determinar el número óptimo de clusters usando método rápido"""
    logger = get_run_logger()
    logger.info("🔍 Determinando número óptimo de clusters...")

    # Usar una muestra representativa para el análisis de clusters óptimos
    sample_size = min(20000, len(data))
    data_sample = data.sample(n=sample_size, random_state=42)
    logger.info(f"📉 Usando muestra de {sample_size} registros para análisis de clusters...")

    # Escalar los datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_sample)

    # Probar valores de k
    k_values = range(3, max_clusters + 1)
    inertia = []

    for k in k_values:
        # Usar MiniBatchKMeans para mayor velocidad
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1000)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        logger.info(f"K={k}: Inercia={kmeans.inertia_:.2f}")

    # Método del codo simplificado - encontrar donde la disminución de inercia se ralentiza
    if len(inertia) > 1:
        reductions = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
        if reductions:
            # Encontrar el punto donde la reducción se hace más pequeña
            reduction_ratios = [reductions[i] / reductions[i-1] if i > 0 else 1 for i in range(len(reductions))]
            optimal_idx = np.argmin(reduction_ratios) + 1 if reduction_ratios else 0
            optimal_k = k_values[optimal_idx]
            logger.info(f"✅ Número óptimo de clusters sugerido: {optimal_k}")
        else:
            optimal_k = 5
            logger.warning(f"⚠️ No se pudo determinar k óptimo, usando valor por defecto: {optimal_k}")
    else:
        optimal_k = 5
        logger.warning(f"⚠️ No se pudo determinar k óptimo, usando valor por defecto: {optimal_k}")

    return optimal_k, scaler

@timing_decorator
@task(name="Realizar clustering K-means optimizado", cache_policy=NO_CACHE)
def perform_clustering(data, n_clusters, scaler):
    # """Realizar clustering K-means optimizado para grandes conjuntos de datos"""
    logger = get_run_logger()
    logger.info(f"🎯 Realizando clustering con {n_clusters} clusters...")

    # Escalar los datos
    scaled_data = scaler.transform(data)

    # Usar MiniBatchKMeans para grandes conjuntos de datos
    logger.info("⚡ Usando MiniBatchKMeans para clustering acelerado...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=5000)
    kmeans.fit(scaled_data)

    # Añadir etiquetas de cluster al DataFrame original
    data['cluster'] = kmeans.labels_

    # Calcular el centroide de cada cluster
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=['lat_center', 'lon_center'])
    centers_df['cluster'] = range(n_clusters)

    # Unir los centroides con los datos
    data = data.merge(centers_df, on='cluster')

    # Calcular la distancia de cada punto a su centroide
    data['distance_to_center'] = np.sqrt(
        (data['lat'] - data['lat_center'])**2 + 
        (data['lon'] - data['lon_center'])**2
    )

    logger.info("✅ Clustering completado")
    return data, kmeans

@timing_decorator
@task(name="Analizar clusters", cache_policy=NO_CACHE)
def analyze_clusters(clustered_data):
    # """Analizar clusters de forma optimizada"""
    logger = get_run_logger()
    logger.info("📊 Analizando clusters...")

    # Estadísticas por cluster
    cluster_stats = clustered_data.groupby('cluster').agg({
        'dr_no': 'count',
        'distance_to_center': ['mean', 'max'],
        'lat': 'mean',
        'lon': 'mean'
    }).round(4)

    cluster_stats.columns = ['num_crimes', 'avg_distance_km', 'max_distance_km', 'center_lat', 'center_lon']
    cluster_stats['avg_distance_km'] = cluster_stats['avg_distance_km'] * 111
    cluster_stats['max_distance_km'] = cluster_stats['max_distance_km'] * 111

    logger.info("\n📈 Estadísticas por cluster:")
    logger.info(f"\n{cluster_stats}")

    # Guardar estadísticas
    cluster_stats.to_csv('cluster_statistics.csv')
    logger.info("✅ Estadísticas de clusters guardadas como 'cluster_statistics.csv'")

    return cluster_stats

@timing_decorator
@task(name="Crear mapa de Los Ángeles con contextily", cache_policy=NO_CACHE)
def create_la_map_with_contextily(clustered_data, cluster_stats):
    # """Crear visualización en mapa de Los Ángeles usando contextily"""
    logger = get_run_logger()
    logger.info("🗺️ Creando mapa de Los Ángeles con clusters...")

    # Usar una muestra para visualización si el dataset es grande
    if len(clustered_data) > VISUALIZATION_SAMPLE:
        logger.info(f"📉 Usando muestra de {VISUALIZATION_SAMPLE} registros para visualización...")
        viz_data = clustered_data.sample(n=VISUALIZATION_SAMPLE, random_state=42)
    else:
        viz_data = clustered_data

    # Convertir a GeoDataFrame
    geometry = [Point(xy) for xy in zip(viz_data['lon'], viz_data['lat'])]
    gdf = gpd.GeoDataFrame(viz_data, geometry=geometry, crs="EPSG:4326")

    # Reproject to Web Mercator for contextily
    gdf = gdf.to_crs(epsg=3857)

    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Plotear los puntos de crimen
    scatter = gdf.plot(ax=ax, column='cluster', categorical=True, 
                      legend=True, markersize=2, alpha=0.6, cmap='tab20')

    # Añadir centroides
    centroids_geometry = [Point(xy) for xy in zip(cluster_stats['center_lon'], cluster_stats['center_lat'])]
    centroids_gdf = gpd.GeoDataFrame(cluster_stats, geometry=centroids_geometry, crs="EPSG:4326")
    centroids_gdf = centroids_gdf.to_crs(epsg=3857)
    centroids_gdf.plot(ax=ax, color='red', marker='X', markersize=100, label='Centroides')

    # Añadir mapa base de Los Ángeles
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Configurar el gráfico
    ax.set_title('Clusters de Criminalidad en Los Ángeles', fontsize=16)
    ax.set_axis_off()

    # Añadir leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Centroides de Cluster'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Incidentes de Crimen')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('la_crime_clusters_map.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Mapa de Los Ángeles con clusters guardado como 'la_crime_clusters_map.png'")
    plt.close()

@timing_decorator
@task(name="Crear mapa de densidad", cache_policy=NO_CACHE)
def create_density_map(clustered_data, cluster_stats):
    # """Crear mapa de densidad en lugar de puntos individuales"""
    logger = get_run_logger()
    logger.info("🗺️ Creando mapa de densidad de clusters...")

    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Crear hexbin plot para mostrar densidad
    hb = ax.hexbin(clustered_data['lon'], clustered_data['lat'], 
                  gridsize=100, cmap='viridis', alpha=0.7, mincnt=1)

    # Añadir centroides
    ax.scatter(cluster_stats['center_lon'], cluster_stats['center_lat'], 
               c='red', marker='X', s=100, label='Centroides')

    # Añadir barra de color
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Densidad de incidentes')

    # Configurar el gráfico
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Densidad de Criminalidad en Los Ángeles', fontsize=16)
    ax.legend()

    plt.tight_layout()
    plt.savefig('la_crime_density_map.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Mapa de densidad guardado como 'la_crime_density_map.png'")
    plt.close()

@timing_decorator
@task(name="Explicar clusters", cache_policy=NO_CACHE)
def explain_clusters(clustered_data, cluster_stats):
    # """
    # Explicación de cómo los clusters ayudan a entender la inseguridad y robos en Los Ángeles.
    # """
    logger = get_run_logger()
    logger.info("\n📝 Explicación interpretativa de los clusters:\n")

    # 1. Número total de clusters
    num_clusters = cluster_stats.shape[0]
    logger.info(f"👉 Se identificaron {num_clusters} clusters principales de criminalidad en Los Ángeles.\n")

    # 2. Relevancia de los centroides
    logger.info("📍 Cada cluster tiene un centroide (zona representativa) que indica un 'hotspot' o punto crítico de inseguridad.")
    logger.info("   Los centroides marcan las áreas donde se concentran los robos y delitos.\n")

    # 3. Estadísticas de inseguridad por cluster
    explanation = []
    for cluster_id, row in cluster_stats.iterrows():
        cluster_info = f"""
   🔹 Cluster {cluster_id}:
      - Total de delitos reportados: {row['num_crimes']:,}
      - Centro aproximado: (lat {row['center_lat']:.4f}, lon {row['center_lon']:.4f})
      - Radio promedio de dispersión: {row['avg_distance_km']:.2f} km
      - Radio máximo de dispersión: {row['max_distance_km']:.2f} km
        """
        explanation.append(cluster_info)
        logger.info(cluster_info)

    # 4. Conexión con percepción de inseguridad
    interpretation = """
📊 Interpretación:
   - Los clusters con más delitos representan zonas de mayor inseguridad.
   - Si un cluster tiene muchos delitos en un área pequeña, es un 'foco rojo'.
   - Los mapas generados muestran estas zonas críticas, lo que permite a ciudadanos y autoridades:
       ✅ Identificar áreas más inseguras
       ✅ Priorizar patrullajes o recursos policiales
       ✅ Prevenir robos al evitar o reforzar esas zonas

💡 En resumen: los clusters convierten miles de incidentes dispersos en un mapa claro de inseguridad,
   mostrando dónde están los principales riesgos de robo en Los Ángeles.
    """
    logger.info(interpretation)

@timing_decorator
@task(name="Visualizar clusters", cache_policy=NO_CACHE)
def visualize_clusters(clustered_data, cluster_stats):
    # """Visualizar clusters en un mapa de Los Ángeles"""
    logger = get_run_logger()
    logger.info("🗺️ Visualizando clusters en el mapa de Los Ángeles...")
    # Aquí iría el código para visualizar los clusters en un mapa
    # Por simplicidad, solo se registrará un mensaje
    logger.info("✅ Visualización de clusters completada")

@timing_decorator
@task(name="Limpiar conexiones", cache_policy=NO_CACHE)
def cleanup_connections(conn, engine):
    # """Limpiar conexiones"""
    logger = get_run_logger()
    try:
        if conn:
            conn.close()
            logger.info("✅ Conexión PostgreSQL cerrada")
        if engine:
            engine.dispose()
            logger.info("✅ Motor SQLAlchemy cerrado")
    except Exception as e:
        logger.error(f"⚠️ Error cerrando conexiones: {e}")




# """
# Script para clustering geográfico con carga por lotes de grandes volúmenes de datos - OPTIMIZADO
# """
# Configuración de optimización para grandes volúmenes - AJUSTADA
BATCH_SIZE = 50000  # Tamaño de lote para carga de datos
MAX_RECORDS = None  # None para todos los registros, o un número para limitar
CLUSTERING_SAMPLE = 100000  # Muestra para clustering si dataset es muy grande

# Coordenadas de Los Ángeles para el mapa
LA_BOUNDS = {
    'min_lat': 33.7,
    'max_lat': 34.4,
    'min_lon': -118.7,
    'max_lon': -118.1
}

@timing_decorator
@task(name="Crear conexión a PostgreSQL", cache_policy=NO_CACHE, retries=3, retry_delay_seconds=10)
def create_connection():
    # """Crear conexión a PostgreSQL"""
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        logger.info("✅ Conexión exitosa a PostgreSQL")
        return conn, engine
    except Exception as e:
        logger.info(f"❌ Error conectando a PostgreSQL: {e}")
        return None, None

@timing_decorator
@task(name="Estimar el número total de registros válidos", cache_policy=NO_CACHE)
def estimate_total_records(conn):
    # """Estimar el número total de registros válidos"""
    logger = get_run_logger()
    try:
        with conn.cursor() as cursor:
            query = f"""
            SELECT COUNT(*) 
            FROM lapd_crime_data 
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            AND lat != 0 AND lon != 0
            AND lat BETWEEN {LA_BOUNDS['min_lat']} AND {LA_BOUNDS['max_lat']}
            AND lon BETWEEN {LA_BOUNDS['min_lon']} AND {LA_BOUNDS['max_lon']};
            """
            cursor.execute(query)
            total_records = cursor.fetchone()[0]
            return total_records
    except Exception as e:
        logger.info(f"❌ Error estimando total de registros: {e}")
        return 0

@timing_decorator
@task(name="Cargar datos geográficos en lotes para manejar grandes volúmenes", cache_policy=NO_CACHE)
def load_geo_data_in_batches(conn, engine, batch_size=BATCH_SIZE, max_records=MAX_RECORDS):
    # """Cargar datos geográficos en lotes para manejar grandes volúmenes"""
    logger = get_run_logger()
    try:
        total_records = estimate_total_records(conn)
        if max_records and max_records < total_records:
            total_records = max_records

        logger.info(f"📊 Total de registros a procesar: {total_records:,}")

        if total_records > CLUSTERING_SAMPLE:
            logger.info(f"⚡ Dataset muy grande, usando muestreo aleatorio de {CLUSTERING_SAMPLE:,} registros")
            query = f"""
            SELECT dr_no, lat, lon, date_occ, crm_cd_desc, area_name
            FROM lapd_crime_data 
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            AND lat != 0 AND lon != 0
            AND lat BETWEEN {LA_BOUNDS['min_lat']} AND {LA_BOUNDS['max_lat']}
            AND lon BETWEEN {LA_BOUNDS['min_lon']} AND {LA_BOUNDS['max_lon']}
            ORDER BY RANDOM()
            LIMIT {CLUSTERING_SAMPLE};
            """
            df = pd.read_sql_query(query, conn)
            logger.info(f"✅ Datos geográficos cargados (muestra): {len(df):,} registros")
            return df

        num_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"📦 Procesando en {num_batches} lotes de {batch_size} registros")

        all_data = []
        for batch_num in range(num_batches):
            offset = batch_num * batch_size
            limit = min(batch_size, total_records - offset)

            query = f"""
            SELECT dr_no, lat, lon, date_occ, crm_cd_desc, area_name
            FROM lapd_crime_data 
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            AND lat != 0 AND lon != 0
            AND lat BETWEEN {LA_BOUNDS['min_lat']} AND {LA_BOUNDS['max_lat']}
            AND lon BETWEEN {LA_BOUNDS['min_lon']} AND {LA_BOUNDS['max_lon']}
            ORDER BY dr_no
            LIMIT {limit} OFFSET {offset};
            """

            logger.info(f"📥 Cargando lote {batch_num + 1}/{num_batches}...")
            df_batch = pd.read_sql_query(query, conn)
            all_data.append(df_batch)
            del df_batch

        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Datos geográficos cargados: {len(df):,} registros")
        return df
    except Exception as e:
        logger.info(f"❌ Error cargando datos geográficos: {e}")
        return pd.DataFrame()

@timing_decorator
@task(name="Determinar el número óptimo de clusters", cache_policy=NO_CACHE)
def determine_optimal_clusters(data, max_clusters=8):
    # """Determinar el número óptimo de clusters usando método rápido"""
    logger = get_run_logger()
    logger.info("🔍 Determinando número óptimo de clusters...")

    sample_size = min(10000, len(data))
    data_sample = data.sample(n=sample_size, random_state=42)
    logger.info(f"📉 Usando muestra de {sample_size} registros para análisis de clusters...")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_sample)

    k_values = range(3, min(max_clusters + 1, 7))
    inertia = []

    for k in k_values:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=1, batch_size=500)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        logger.info(f"K={k}: Inercia={kmeans.inertia_:.2f}")

    if len(inertia) > 1:
        reductions = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
        if reductions:
            reduction_ratios = [reductions[i] / reductions[i-1] if i > 0 and reductions[i-1] != 0 else 1 for i in range(len(reductions))]
            optimal_idx = np.argmin(reduction_ratios) + 1 if reduction_ratios else 0
            optimal_k = k_values[min(optimal_idx, len(k_values)-1)]
            logger.info(f"✅ Número óptimo de clusters sugerido: {optimal_k}")
        else:
            optimal_k = 4
            logger.info(f"⚠️ No se pudo determinar k óptimo, usando valor por defecto: {optimal_k}")
    else:
        optimal_k = 4
        logger.info(f"⚠️ No se pudo determinar k óptimo, usando valor por defecto: {optimal_k}")

    return optimal_k, scaler

@timing_decorator
@task(name="Realizar clustering K-means optimizado", cache_policy=NO_CACHE)
def perform_clustering(data, n_clusters, scaler):
    logger = get_run_logger()
    logger.info(f"🎯 Realizando clustering con {n_clusters} clusters...")

    # Solo usamos lat y lon
    features = ['lat', 'lon']
    scaled_data = scaler.fit_transform(data[features])

    logger.info("⚡ Usando MiniBatchKMeans para clustering acelerado...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=1,
        batch_size=2000,
        max_iter=10000
    )
    kmeans.fit(scaled_data)

    # Guardar etiquetas en copia del dataframe
    data_copy = data.copy()
    data_copy['cluster'] = kmeans.labels_

    # Obtener centros (en escala original)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=['lat_center', 'lon_center'])
    centers_df['cluster'] = range(n_clusters)

    # Unir centros al dataset
    data_copy = data_copy.merge(centers_df, on='cluster')

    # Calcular distancia al centro del cluster
    data_copy['distance_to_center'] = np.sqrt(
        (data_copy['lat'] - data_copy['lat_center'])**2 + 
        (data_copy['lon'] - data_copy['lon_center'])**2
    )

    logger.info("✅ Clustering completado")
    return data_copy, kmeans


@timing_decorator
@task(name="Analizar clusters", cache_policy=NO_CACHE)
def analyze_clusters(clustered_data):
    # """Analizar clusters de forma optimizada"""
    logger = get_run_logger()
    logger.info("📊 Analizando clusters...")

    cluster_stats = clustered_data.groupby('cluster').agg({
        'dr_no': 'count',
        'distance_to_center': ['mean', 'max'],
        'lat': 'mean',
        'lon': 'mean'
    }).round(4)

    cluster_stats.columns = ['num_crimes', 'avg_distance_km', 'max_distance_km', 'center_lat', 'center_lon']
    cluster_stats['avg_distance_km'] = cluster_stats['avg_distance_km'] * 111
    cluster_stats['max_distance_km'] = cluster_stats['max_distance_km'] * 111

    logger.info("\n📈 Estadísticas por cluster:")
    logger.info(cluster_stats)

    cluster_stats.to_csv('cluster_statistics.csv')
    logger.info("✅ Estadísticas de clusters guardadas como 'cluster_statistics.csv'")

    return cluster_stats

@timing_decorator
@task(name="Crear mapa interactivo de clústeres", cache_policy=NO_CACHE)
def create_interactive_map(clustered_data):
    # """Mapa interactivo de clústeres con Folium - OPTIMIZADO"""
    logger = get_run_logger()
    try:
        import folium
        logger.info("🗺️ Creando mapa interactivo...")

        sample_size = min(1000, len(clustered_data))
        sample_data = clustered_data.sample(sample_size, random_state=42)
        logger.info(f"📉 Usando muestra de {sample_size} registros para mapa interactivo...")

        m = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles="cartodbpositron")

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige']

        for _, row in sample_data.iterrows():
            color = colors[int(row['cluster']) % len(colors)]
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.5,
                popup=f"Cluster {row['cluster']} - {row.get('crm_cd_desc', 'N/A')}"
            ).add_to(m)

        m.save("la_crime_clusters_interactive.html")
        logger.info("✅ Mapa interactivo guardado como 'la_crime_clusters_interactive.html'")
    except ImportError:
        logger.info("⚠️ Folium no está disponible, saltando mapa interactivo")
    except Exception as e:
        logger.info(f"⚠️ Error creando mapa interactivo: {e}")

@timing_decorator
@task(name="Analizar tendencias temporales", cache_policy=NO_CACHE)
def analyze_temporal_trends(df):
    # """Analizar evolución temporal de crímenes - OPTIMIZADO"""
    logger = get_run_logger()
    try:
        logger.info("📈 Analizando tendencias temporales...")
        df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')

        valid_dates = df.dropna(subset=['date_occ'])
        if valid_dates.empty:
            logger.info("⚠️ No hay fechas válidas para análisis temporal")
            return

        crimes_per_year = valid_dates.groupby(valid_dates['date_occ'].dt.year).size()

        if crimes_per_year.empty:
            logger.info("⚠️ No hay datos temporales para graficar")
            return

        plt.figure(figsize=(8, 4))
        crimes_per_year.plot(kind='bar')
        plt.title("Evolución de crímenes por año en Los Ángeles")
        plt.xlabel("Año")
        plt.ylabel("Número de incidentes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("crime_trends_by_year.png", dpi=100)
        plt.close()
        logger.info("✅ Tendencias temporales guardadas en 'crime_trends_by_year.png'")
    except Exception as e:
        logger.info(f"⚠️ Error en análisis temporal: {e}")

@timing_decorator
@task(name="Explicar clusters", cache_policy=NO_CACHE)
def explain_clusters(clustered_data, cluster_stats):
    # """Explicación de cómo los clusters ayudan a entender la inseguridad y robos en Los Ángeles."""
    logger = get_run_logger()
    logger.info("\n📝 Explicación interpretativa de los clusters:\n")

    num_clusters = cluster_stats.shape[0]
    logger.info(f"👉 Se identificaron {num_clusters} clusters principales de criminalidad en Los Ángeles.\n")

    logger.info("📍 Cada cluster tiene un centroide (zona representativa) que indica un 'hotspot' o punto crítico de inseguridad.")
    logger.info("   Los centroides marcan las áreas donde se concentran los robos y delitos.\n")

    for cluster_id, row in cluster_stats.iterrows():
        logger.info(f"   🔹 Cluster {cluster_id}:")
        logger.info(f"      - Total de delitos reportados: {row['num_crimes']:,}")
        logger.info(f"      - Centro aproximado: (lat {row['center_lat']:.4f}, lon {row['center_lon']:.4f})")
        logger.info(f"      - Radio promedio de dispersión: {row['avg_distance_km']:.2f} km")
        logger.info(f"      - Radio máximo de dispersión: {row['max_distance_km']:.2f} km")
        print()

    logger.info("📊 Interpretación:")
    logger.info("   - Los clusters con más delitos representan zonas de mayor inseguridad.")
    logger.info("   - Si un cluster tiene muchos delitos en un área pequeña, es un 'foco rojo'.")
    logger.info("   - El mapa interactivo generado muestra estas zonas críticas, lo que permite a ciudadanos y autoridades:")
    logger.info("       ✅ Identificar áreas más inseguras")
    logger.info("       ✅ Priorizar patrullajes o recursos policiales")
    logger.info("       ✅ Prevenir robos al evitar o reforzar esas zonas\n")

    logger.info("💡 En resumen: los clusters convierten miles de incidentes dispersos en un mapa claro de inseguridad,")
    logger.info("   mostrando dónde están los principales riesgos de robo en Los Ángeles.\n")

@timing_decorator
@task(name="Explicar tendencias temporales y espaciales", cache_policy=NO_CACHE)
def explain_temporal_and_spatial(cluster_stats, geo_data):
    # """Explicación de tendencias temporales y espaciales"""
    logger = get_run_logger()
    logger.info("\n📖 Interpretación ampliada:\n")
    logger.info("🔎 Temporal:")
    logger.info("   - Si observamos la evolución por año o mes, podemos ver si la inseguridad está aumentando o bajando.")
    logger.info("   - Si un clúster muestra picos en horarios específicos (ej. de noche), puede indicar problemas de patrullaje o iluminación.")

    logger.info("\n🌍 Espacial:")
    logger.info("   - Los clústeres confirman que la inseguridad se concentra en puntos concretos, no de forma aleatoria.")
    logger.info("   - Esto refuerza la idea de que los 'puntos calientes' requieren atención prioritaria.")

@timing_decorator
@task(name="Validar clusters", cache_policy=NO_CACHE)
def validate_clusters(data, labels):
    # """Validar calidad de los clusters con métricas de separación"""
    logger = get_run_logger()
    try:
        from sklearn.metrics import silhouette_score
        if len(data) > 20000:
            sample_idx = np.random.choice(len(data), 20000, replace=False)
            data_sample = data.iloc[sample_idx]
            labels_sample = labels[sample_idx]
        else:
            data_sample = data
            labels_sample = labels

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_sample)
        score = silhouette_score(scaled_data, labels_sample)
        logger.info(f"📏 Silhouette Score: {score:.3f} (cercano a 1 es mejor separación)")
    except Exception as e:
        logger.info(f"⚠️ No se pudo calcular Silhouette Score: {e}")

@timing_decorator
@task(name="Preparar características para clustering", cache_policy=NO_CACHE)
def prepare_features(df):
    # """Crear variables adicionales para clustering - OPTIMIZADO"""
    logger = get_run_logger()
    logger.info("🔧 Preparando features para clustering...")

    df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
    df['year'] = df['date_occ'].dt.year
    df['month'] = df['date_occ'].dt.month
    df['hour'] = df['date_occ'].dt.hour

    crime_mapping = {crime: idx for idx, crime in enumerate(df['crm_cd_desc'].unique())}
    df['crime_type'] = df['crm_cd_desc'].map(crime_mapping)

    df['hour'] = df['hour'].fillna(12)
    df['month'] = df['month'].fillna(6)
    df['crime_type'] = df['crime_type'].fillna(0)

    features = df[['lat', 'lon', 'hour', 'month', 'crime_type']].copy()
    return features


@timing_decorator
@task(name="Geographic Crime Clustering Analysis - Optimized")
def geographic_clustering_flow():
    # """Función principal optimizada para grandes volúmenes de datos"""
    logger = get_run_logger()
    logger.info("🚀 Iniciando clustering geográfico con carga por lotes - VERSIÓN OPTIMIZADA")
    logger.info("="*80)

    conn, engine = create_connection()
    if conn is None or engine is None:
        return

    try:
        geo_data = load_geo_data_in_batches(conn, engine, batch_size=BATCH_SIZE, max_records=MAX_RECORDS)

        if geo_data.empty:
            logger.info("❌ No hay datos geográficos para analizar")
            return

        logger.info(f"📊 Trabajando con {len(geo_data):,} registros")

        analyze_temporal_trends(geo_data)

        clustering_data = prepare_features(geo_data)

        optimal_k, scaler = determine_optimal_clusters(clustering_data)

        clustered_data, kmeans = perform_clustering(clustering_data, optimal_k, scaler)
        validate_clusters(clustering_data, clustered_data['cluster'].values)

        result_data = pd.concat([
            geo_data.reset_index(drop=True),
            clustered_data[['cluster', 'lat_center', 'lon_center', 'distance_to_center']].reset_index(drop=True)
        ], axis=1)

        cluster_stats = analyze_clusters(result_data)

        logger.info("\n🎨 Generando visualización interactiva...")
        create_interactive_map(result_data)

        logger.info("\n🎉 Análisis de clustering completado!")
        logger.info("\n📋 Archivos generados:")
        logger.info("   - cluster_statistics.csv: Estadísticas de cada cluster")
        logger.info("   - crime_trends_by_year.png: Tendencias temporales")
        logger.info("   - la_crime_clusters_interactive.html: Mapa interactivo")

        explain_clusters(result_data, cluster_stats)
        explain_temporal_and_spatial(cluster_stats, geo_data)

    except Exception as e:
        logger.info(f"❌ Error durante el clustering: {e}")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
        if engine:
            engine.dispose()




# ### Heatmap interactivo
# 

@timing_decorator
@task(name="Heatmap",cache_policy=NO_CACHE)
def Heatmap():
    conn, engine = create_connection()
    if conn is None or engine is None:
        return

    # Reemplazar parquet por consulta SQL
    query = "SELECT * FROM lapd_crime_data;"  # Ajusta al nombre real de tu tabla
    df = pd.read_sql(query, conn)

    # Revisar columnas disponibles
    print(df.columns)

    import folium
    from folium.plugins import HeatMap

    # Filtrar registros con coordenadas válidas
    df_map = df[['lat', 'lon']].dropna()

    # Crear mapa base centrado en Los Ángeles
    m = folium.Map(location=[34.05, -118.25], zoom_start=10)

    # Agregar capa de calor
    HeatMap(df_map.values, radius=10, blur=15, max_zoom=1).add_to(m)

    # Guardar en HTML
    m.save("heatmap_interactivo.html")
    m



# ### ¿Existen diferencias significativas en la edad promedio de las víctimas según la zona geográfica y el tipo de delito?

# Preparación de datos


@timing_decorator
@task(name="Query Data", cache_policy=NO_CACHE)
def query_data():
    # Consultar columnas necesarias
    query = """
    SELECT vict_age, area_name, crm_cd_desc
    FROM lapd_crime_data
    WHERE vict_age IS NOT NULL
    AND vict_age ~ '^[0-9]+$'  -- solo valores numéricos
    AND area_name IS NOT NULL
    AND crm_cd_desc IS NOT NULL;
    """

    print("📥 Consultando datos desde PostgreSQL...")
    # Crear conexión con SQLAlchemy
    engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

    df = pd.read_sql(query, engine)

    # Asegurar que la columna edad es numérica
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')

    # Eliminar edades inválidas (ejemplo: 0 o negativas, mayores de 100)
    df = df[(df['vict_age'] > 0) & (df['vict_age'] < 100)]

    print(f"✅ Registros válidos: {len(df):,}")
    print(df.head())



# ### ANOVA de dos factores (Zona + Tipo de delito)
# 
# Si el p-value < 0.05, hay diferencias significativas.


@timing_decorator
@task(name="Filter Top Crimes", cache_policy=NO_CACHE)
def filter_top_crimes():
    # Crear conexión con SQLAlchemy
    conn, engine = create_connection()
    if conn is None or engine is None:
        return

    # Reemplazar parquet por consulta SQL
    query = """
    SELECT vict_age, area_name, crm_cd_desc
    FROM lapd_crime_data
    WHERE vict_age IS NOT NULL
    AND vict_age ~ '^[0-9]+$'   -- solo valores numéricos
    AND CAST(vict_age AS INTEGER) > 0
    AND CAST(vict_age AS INTEGER) < 100
    AND area_name IS NOT NULL
    AND crm_cd_desc IS NOT NULL;
    """

    df = pd.read_sql(query, conn)

    # 1. Filtrar top 10 delitos
    top_crimes = df['crm_cd_desc'].value_counts().head(10).index
    df_filtered = df[df['crm_cd_desc'].isin(top_crimes)]

    # 2. Tomar una muestra
    df_sample = df_filtered.sample(min(50000, len(df_filtered)), random_state=42)

    # 3. Asegurar tipo numérico en la muestra
    df_sample['vict_age'] = pd.to_numeric(df_sample['vict_age'], errors='coerce')
    df_sample = df_sample.dropna(subset=['vict_age'])
    df_sample = df_sample[(df_sample['vict_age'] > 0) & (df_sample['vict_age'] < 100)]

    print(df_sample['vict_age'].dtype)   # debería ser int64 o float64
    print(df_sample['vict_age'].unique()[:20])  # ejemplos

    # 4. Modelo ANOVA
    model = ols('vict_age ~ C(area_name) + C(crm_cd_desc)', data=df_sample).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

# ### Visualización

@timing_decorator
@task(name="Visualize Avg Age by Crime and Area", cache_policy=NO_CACHE)
def visualize_avg_age_by_crime_and_area():
     # Consultar columnas necesarias
    query = """
    SELECT vict_age, area_name, crm_cd_desc
    FROM lapd_crime_data
    WHERE vict_age IS NOT NULL
    AND vict_age ~ '^[0-9]+$'  -- solo valores numéricos
    AND area_name IS NOT NULL
    AND crm_cd_desc IS NOT NULL;
    """

    print("📥 Consultando datos desde PostgreSQL...")
    # Crear conexión con SQLAlchemy
    engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

    df = pd.read_sql(query, engine)
    sns.set(style="whitegrid")
    # 1. Contar los delitos más frecuentes
    top10_delitos = df["crm_cd_desc"].value_counts().head(10).index

    # 2. Filtrar solo esos delitos en el dataframe
    df_top10 = df[df["crm_cd_desc"].isin(top10_delitos)]

    # 3. Filtrar solo 5 zonas de interés
    zonas_seleccionadas = ["Hollywood", "Central", "Southwest", "Pacific", "77th Street"]  # <-- cámbialas como prefieras
    df_top10 = df_top10[df_top10["area_name"].isin(zonas_seleccionadas)]

    # 🔑 Asegurar tipo numérico para vict_age
    df_top10["vict_age"] = pd.to_numeric(df_top10["vict_age"], errors="coerce")
    df_top10 = df_top10.dropna(subset=["vict_age"])
    df_top10 = df_top10[(df_top10["vict_age"] > 0) & (df_top10["vict_age"] < 100)]
    # 4. Calcular promedios por zona y delito
    avg_age = df_top10.groupby(["crm_cd_desc", "area_name"])["vict_age"].mean().reset_index()

    # 5. Visualización
    plt.figure(figsize=(14,7))
    sns.barplot(
        data=avg_age,
        x="crm_cd_desc",
        y="vict_age",
        hue="area_name"
    )

    # Mejorar la estética
    plt.xticks(rotation=80, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.xlabel("Tipo de delito", fontsize=12)
    plt.ylabel("Edad promedio de las víctimas", fontsize=12)
    plt.title("Edad promedio de las víctimas por delito y zona (5 zonas seleccionadas)", fontsize=14, weight="bold")
    plt.legend(title="Zona geográfica", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

@flow(name="Flujo principal de carga LAPD")
def main():
    # """Función principal"""
    logger = get_run_logger()
    logger.info("🚀 Iniciando carga de datos LAPD desde Parquet a PostgreSQL")
    logger.info(f"📁 Archivo: {PARQUET_FILE_PATH}")
    logger.info("="*80)

    # Verificar conexión con Docker
    logger.info("🔍 Verificando conexión a PostgreSQL...")

    success = load_parquet_to_postgres()  # Lotes más pequeños por seguridad

    if success:
        logger.info("\n🎉 ¡Carga completada exitosamente!")
        logger.info("\nPuedes conectarte a la base" 
        " de datos y hacer consultas:")
        logger.info("docker exec -it postgres16_bigdata psql -U psqluser -d BigdataFinal")
    else:
        logger.info("\n❌ La carga falló. Revisa los errores anteriores.")

    logger.info("🔍 Iniciando análisis de missing values en la tabla lapd_crime_data")
    logger.info("="*80)
    analyze_missing_values()

    logger.info("🚀 Iniciando limpieza de datos en la tabla lapd_crime_data")
    logger.info("="*80)

    conn, engine = create_connection()  # Desempaquetar la tupla
    if conn is None or engine is None:
        return

    try:
        # Paso 0: Limpiar valores especiales (0 en coordenadas, valores inválidos en vict_age)
        clean_special_values(conn)

        # Paso 1: Eliminar columnas
        drop_columns(conn)

        # Paso 2: Eliminar filas con nulos en columnas clave
        delete_rows_with_nulls(conn)

        # Paso 3: Imputar valores
        impute_vict_sex(conn)
        impute_vict_descent(conn)
        impute_vict_age(conn)

        logger.info("🎉 Limpieza completada exitosamente!")

    except Exception as e:
        logger.info(f"❌ Error durante la limpieza: {e}")
    finally:
        if conn:
            conn.close()  # Cerrar la conexión psycopg2
        if engine:
            engine.dispose()  # Desechar el motor de SQLAlchemy

    logger.info("🔍 Iniciando análisis de missing values en la tabla lapd_crime_data")
    logger.info("="*80)
    analyze_missing_values(GraficPath='missing_values_analysi_cleaned.png')

    logger.info("🔍 Iniciando análisis de outliers con boxplots")
    logger.info("="*80)

    conn, engine = create_connection()
    if conn is None or engine is None:
        return

    try:
        # Obtener columnas numéricas
        numeric_columns = get_numeric_columns(conn)

        if not numeric_columns:
            logger.info("❌ No se encontraron columnas numéricas en la tabla")
            return

        logger.info(f"📋 Columnas numéricas encontradas: {', '.join(numeric_columns)}")

        # Cargar datos numéricos
        df = load_numeric_data(engine, numeric_columns)

        if df.empty:
            logger.info("❌ No se pudieron cargar los datos")
            return

        logger.info(f"📊 Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas numéricas")

        # Analizar outliers
        analyze_outliers(df)

        logger.info("\n🎉 Análisis de outliers completado!")

    except Exception as e:
        logger.info(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.logger.info_exc()
    finally:
        if conn:
            conn.close()  # Cerrar la conexión psycopg2
        if engine:
            engine.dispose()  # Desechar el motor de SQLAlchemy

    logger.info("🚀 Iniciando clustering geográfico con carga por lotes usando Prefect")
    logger.info("="*80)

    conn = None
    engine = None

    try:
        # Task 1: Crear conexión
        conn, engine = create_connection()

        # Task 2: Estimar total de registros
        total_records = estimate_total_records(conn)

        if total_records == 0:
            logger.error("❌ No hay datos geográficos para analizar")
            return

        # Task 3: Cargar datos en lotes
        geo_data = load_geo_data_in_batches(conn, engine, batch_size=BATCH_SIZE, max_records=MAX_RECORDS)

        if geo_data.empty:
            logger.error("❌ No se pudieron cargar los datos geográficos")
            return

        # Task 4: Preparar datos para clustering (solo lat y lon)
        clustering_data = geo_data[['lat', 'lon']].copy()

        # Task 5: Determinar número óptimo de clusters
        optimal_k, scaler = determine_optimal_clusters(clustering_data)

        # Task 6: Realizar clustering con todos los datos
        clustered_data, kmeans = perform_clustering(clustering_data, optimal_k, scaler)

        # Task 7: Añadir información adicional al resultado
        clustered_data = pd.concat([
            geo_data.reset_index(drop=True),
            clustered_data[['cluster', 'lat_center', 'lon_center', 'distance_to_center']]
        ], axis=1)

        # Task 8: Analizar clusters
        cluster_stats = analyze_clusters(clustered_data)

        # Task 9: Visualizar clusters en mapa de Los Ángeles
        create_la_map_with_contextily(clustered_data, cluster_stats)

        # Task 10: Crear mapa de densidad
        create_density_map(clustered_data, cluster_stats)

        # Task 11: Ejecutar explicación interpretativa
        explain_clusters(clustered_data, cluster_stats)

        logger.info("\n🎉 Análisis de clustering completado!")
        logger.info("\n📋 Archivos generados:")
        logger.info("   - cluster_statistics.csv: Estadísticas de cada cluster")
        logger.info("   - la_crime_clusters_map.png: Mapa estático de clusters en Los Ángeles")
        logger.info("   - la_crime_density_map.png: Mapa de densidad de criminalidad")

        return {
            "clustered_data": clustered_data,
            "cluster_stats": cluster_stats,
            "kmeans_model": kmeans,
            "optimal_k": optimal_k
        }

    except Exception as e:
        logger.error(f"❌ Error durante el clustering: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Task 12: Limpiar conexiones siempre al final
        if conn is not None or engine is not None:
            cleanup_connections(conn, engine)

if __name__ == "__main__":
    # Ejecutar el flow
    main()
    geographic_clustering_flow()
    Heatmap()
    query_data()
    filter_top_crimes()
    visualize_avg_age_by_crime_and_area()



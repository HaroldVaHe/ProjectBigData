import os
import requests
import pandas as pd
import json

url_base = "https://data.lacity.org/resource/2nrs-mtv8.json"

# Definir tamaño del bloque
chunk_size = 100000
offset = 0
all_data = []

while True:
    url = f"{url_base}?$limit={chunk_size}&$offset={offset}"
    print(f"Descargando desde offset {offset}...")

    response = requests.get(url)
    data = response.json()

    if not data:  # Si ya no hay más datos, se corta
        break

    all_data.extend(data)
    offset += chunk_size
    print(f"Acumulados: {len(all_data)} registros")

# Pasar a DataFrame
df = pd.DataFrame(all_data)

# Guardar en JSON (en formato lista de registros)
with open("Crime_Data_from_2020_to_Present.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False)

# Guardar en CSV y parquet
df.to_csv("Crime_Data_from_2020_to_Present.csv", index=False)
df.to_parquet("Crime_Data_from_2020_to_Present.parquet", engine="pyarrow", index=False)

# Confirmaciones
print("Total registros:", df.shape)
print(df.head())


# Rutas de los archivos
json_file = "Crime_Data_from_2020_to_Present.json"
csv_file = "Crime_Data_from_2020_to_Present.csv"
parquet_file = "Crime_Data_from_2020_to_Present.parquet" # salida en formato parquet


# Leer JSON, CSV y Parquet
df_json = pd.read_json(json_file)
df_csv = pd.read_csv(csv_file)
df_parquet = pd.read_parquet(parquet_file)

# Mostrar información del JSON
print("=== JSON ===")
print(df_json.head())
print("Shape JSON:", df_json.shape)

# Mostrar información del CSV
print("\n=== CSV ===")
print(df_csv.head())
print("Shape CSV:", df_csv.shape)

# Mostrar información del Parquet
print("\n=== Parquet ===")
print(df_parquet.head())
print("Shape Parquet:", df_parquet.shape)

# Comparar cantidad de filas y columnas
if df_json.shape == df_csv.shape == df_parquet.shape:
    print("\n✅ Todos los archivos tienen la misma cantidad de datos.")
else:
    print("\n⚠️ Los archivos tienen diferente cantidad de datos.")

# Comparar tamaños en disco
size_json = os.path.getsize(json_file) / (1024 * 1024)
size_csv = os.path.getsize(csv_file) / (1024 * 1024)
size_parquet = os.path.getsize(parquet_file) / (1024 * 1024)

print("\n=== Comparación de tamaños (MB) ===")
print(f"JSON:    {size_json:.2f} MB")
print(f"CSV:     {size_csv:.2f} MB")
print(f"Parquet: {size_parquet:.2f} MB")
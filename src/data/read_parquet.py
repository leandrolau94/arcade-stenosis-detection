import os
import sys
from pathlib import Path

# Configuración de PySpark en Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spark"
    / "stenosis_features"
)


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("ARCADE-Read-Parquet")
    .master("local[*]")
    .getOrCreate()
)

print(f"Spark version: {spark.version}")
print(f"Leyendo Parquet desde: {PARQUET_PATH}")


# ============================================================
# COMPROBAR QUE EXISTE
# ============================================================

if not PARQUET_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el directorio Parquet: {PARQUET_PATH}"
    )

print("✓ Directorio Parquet encontrado")


# ============================================================
# LEER PARQUET
# ============================================================

df = spark.read.parquet(str(PARQUET_PATH))

print("\n✓ Parquet leído correctamente")


# ============================================================
# INFORMACIÓN DEL DATAFRAME
# ============================================================

print("\nNúmero de registros:")
print(df.count())

print("\nColumnas:")
print(df.columns)

print("\nSchema:")
df.printSchema()


# ============================================================
# PRIMERAS FILAS
# ============================================================

print("\nPrimeras 10 lesiones:")
df.show(10, truncate=False)


# ============================================================
# RESUMEN
# ============================================================

print("\nResumen estadístico:")
df.describe().show()


# ============================================================
# CERRAR SPARK
# ============================================================

spark.stop()

print("\nSparkSession cerrada correctamente.")
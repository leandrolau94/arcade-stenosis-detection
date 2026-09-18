import os
import sys
from pathlib import Path

# PySpark en WSL
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession


# ============================================================
# CONFIGURACIÓN
# ============================================================

PARQUET_PATH = (Path.home() / "arcade-spark-data" / "stenosis_features")


# ============================================================
# CREAR SPARK SESSION
# ============================================================

spark = (SparkSession.builder.appName("ARCADE-Spark-SQL").master("local[*]").getOrCreate())

print(f"Spark version: {spark.version}")
print(f"Leyendo Parquet desde: {PARQUET_PATH}")


# ============================================================
# LEER PARQUET
# ============================================================

df = spark.read.parquet(str(PARQUET_PATH))

print("\n✓ Parquet leído correctamente")
print(f"Número de lesiones: {df.count()}")


# ============================================================
# CREAR VISTA SQL
# ============================================================

df.createOrReplaceTempView("stenosis")

print("✓ Vista SQL 'stenosis' creada")


# ============================================================
# PRIMERA CONSULTA SQL
# ============================================================

print("\n=== PRIMERA CONSULTA SQL ===")

result = spark.sql("""SELECT annotation_id, image_id, area, lesion_width, lesion_height, aspect_ratio, relative_area FROM stenosis LIMIT 10""")

result.show(truncate=False)

print("\n=== ESTENOSIS POR IMAGEN ===")

result1 = spark.sql("""SELECT image_id, COUNT(*) AS num_stenoses FROM stenosis GROUP BY image_id ORDER BY num_stenoses DESC""")

result1.show(20, truncate=False)

print("\n=== CARACTERÍSTICAS SEGÚN NÚMERO DE ESTENOSIS ===")

result2 = spark.sql("""
    SELECT
        image_id,
        COUNT(*) AS num_stenoses,
        ROUND(AVG(area), 2) AS avg_area,
        ROUND(AVG(lesion_width), 2) AS avg_width,
        ROUND(AVG(lesion_height), 2) AS avg_height,
        ROUND(AVG(aspect_ratio), 2) AS avg_aspect_ratio,
        ROUND(AVG(relative_area), 4) AS avg_relative_area
    FROM stenosis
    GROUP BY image_id
    ORDER BY num_stenoses DESC
""")

result2.show(20, truncate=False)

print("\n=== RESUMEN SEGÚN NÚMERO DE ESTENOSIS ===")

result3 = spark.sql("""
    WITH image_stats AS (
        SELECT
            image_id,
            COUNT(*) AS num_stenoses,
            AVG(area) AS avg_area,
            AVG(lesion_width) AS avg_width,
            AVG(lesion_height) AS avg_height
        FROM stenosis
        GROUP BY image_id
    )

    SELECT
        num_stenoses,
        COUNT(*) AS num_images,
        ROUND(AVG(avg_area), 2) AS avg_area,
        ROUND(AVG(avg_width), 2) AS avg_width,
        ROUND(AVG(avg_height), 2) AS avg_height
    FROM image_stats
    GROUP BY num_stenoses
    ORDER BY num_stenoses
""")

result3.show(truncate=False)

# ============================================================
# 5. DATASET FINAL: ESTADÍSTICAS POR IMAGEN
# ============================================================

result4 = spark.sql("""
    SELECT
        image_id,
        COUNT(*) AS num_stenoses,
        ROUND(SUM(area), 2) AS total_area,
        ROUND(AVG(area), 2) AS avg_area,
        ROUND(MAX(area), 2) AS max_area,
        ROUND(AVG(lesion_width), 2) AS avg_width,
        ROUND(AVG(lesion_height), 2) AS avg_height,
        ROUND(AVG(aspect_ratio), 2) AS avg_aspect_ratio,
        ROUND(AVG(relative_area), 4) AS avg_relative_area
    FROM stenosis
    GROUP BY image_id
    ORDER BY image_id
""")

print("\n=== ESTADÍSTICAS FINALES POR IMAGEN ===")
result4.show(20, truncate=False)

print(f"\nNúmero de imágenes con estenosis: {result4.count()}")


# ============================================================
# 6. EXPORTAR A PARQUET
# ============================================================

OUTPUT_IMAGE_STATS = (
    Path.home()
    / "arcade-spark-data"
    / "image_statistics"
)

result4.write \
    .mode("overwrite") \
    .parquet(str(OUTPUT_IMAGE_STATS))

print(f"\n✓ Dataset de estadísticas guardado en:")
print(OUTPUT_IMAGE_STATS)

# ============================================================
# CERRAR SPARK
# ============================================================

spark.stop()

print("\nSparkSession cerrada correctamente.")
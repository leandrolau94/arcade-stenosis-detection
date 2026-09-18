import os
import json
from pathlib import Path

# Forzar a PySpark a utilizar el Python del entorno virtual actual
os.environ["PYSPARK_PYTHON"] = os.sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = os.sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import coalesce, lit, col, when

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = (
    PROJECT_ROOT
    / "arcade"
    / "stenosis"
    / "train"
    / "annotations"
    / "train.json"
)


# ============================================================
# 2. CREAR SPARK SESSION
# ============================================================

spark = (SparkSession.builder.appName("ARCADE-Stenosis-EDA").master("local[*]").getOrCreate())

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("ARCADE - PySpark EDA")
print("=" * 60)
print(f"Spark version: {spark.version}")
print(f"JSON: {TRAIN_JSON}")


# ============================================================
# 3. COMPROBAR QUE EXISTE train.json
# ============================================================

if not TRAIN_JSON.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo:\n{TRAIN_JSON}"
    )

print("✓ train.json encontrado")


# ============================================================
# 4. LEER EL JSON
# ============================================================

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    arcade_data = json.load(f)

print("\nClaves del JSON:")
print(arcade_data.keys())


# ============================================================
# 5. CREAR DATAFRAMES DE SPARK
# ============================================================

images_df = spark.createDataFrame(arcade_data["images"])
annotations_df = spark.createDataFrame(arcade_data["annotations"])


# ============================================================
# 6. MOSTRAR ESQUEMAS
# ============================================================

print("\n" + "=" * 60)
print("SCHEMA - IMAGES")
print("=" * 60)

images_df.printSchema()

print("\n" + "=" * 60)
print("SCHEMA - ANNOTATIONS")
print("=" * 60)

annotations_df.printSchema()


# ============================================================
# 7. CONTAR REGISTROS
# ============================================================

print("\n" + "=" * 60)
print("COUNTS")
print("=" * 60)

print(f"Images:       {images_df.count()}")
print(f"Annotations:  {annotations_df.count()}")


# ============================================================
# 8. MOSTRAR EJEMPLOS
# ============================================================

print("\n" + "=" * 60)
print("PRIMERAS IMÁGENES")
print("=" * 60)

images_df.show(5, truncate=False)

print("\n" + "=" * 60)
print("PRIMERAS ANOTACIONES")
print("=" * 60)

annotations_df.show(5, truncate=False)

# ============================================================
# ESTENOSIS POR IMAGEN
# ============================================================

print("=" * 60)
print("ESTENOSIS POR IMAGEN")
print("=" * 60)

# Nos quedamos únicamente con la categoría de estenosis
stenosis_df = annotations_df.filter(annotations_df.category_id == 26)

print("Anotaciones de estenosis:", stenosis_df.count())

# Contamos cuántas estenosis tiene cada imagen
stenosis_per_image = (stenosis_df.groupBy("image_id").count().withColumnRenamed("count", "num_stenoses"))

stenosis_per_image.show(10)

# ============================================================
# JOIN: IMÁGENES + ESTENOSIS
# ============================================================

print("=" * 60)
print("IMÁGENES + ESTENOSIS")
print("=" * 60)

image_stenosis_df = (
    images_df.join(
        stenosis_per_image,
        images_df.id == stenosis_per_image.image_id,
        "left"
    ).select(
        images_df.id.alias("image_id"),
        images_df.file_name,
        images_df.width,
        images_df.height,
        coalesce(stenosis_per_image.num_stenoses, lit(0)).alias("num_stenoses")
    )
)

image_stenosis_df.show(10)

# ============================================================
# DISTRIBUCIÓN DE ESTENOSIS
# ============================================================

print("=" * 60)
print("DISTRIBUCIÓN DE ESTENOSIS POR IMAGEN")
print("=" * 60)

distribution_df = (image_stenosis_df.groupBy("num_stenoses").count().orderBy("num_stenoses"))

distribution_df.show()

# ============================================================
# FEATURE ENGINEERING - LESIONES
# ============================================================

print("=" * 60)
print("FEATURE ENGINEERING - LESIONES")
print("=" * 60)

lesion_features_df = (
    annotations_df
    .filter(col("category_id") == 26)
    .select(
        col("id").alias("annotation_id"),
        col("image_id"),
        col("area"),
        col("bbox")
    )
    .withColumn("x", col("bbox")[0])
    .withColumn("y", col("bbox")[1])
    .withColumn("lesion_width", col("bbox")[2])
    .withColumn("lesion_height", col("bbox")[3])
)

lesion_features_df.select(
    "annotation_id",
    "image_id",
    "area",
    "x",
    "y",
    "lesion_width",
    "lesion_height"
).show(10)

lesion_features_df = lesion_features_df.withColumn(
    "aspect_ratio", when(
        col("lesion_height") > 0,
        col("lesion_width") / col("lesion_height")
    ).otherwise(None)
)

lesion_features_df = lesion_features_df.withColumn(
    "bbox_area",
    col("lesion_width") * col("lesion_height")
)

lesion_features_df.select(
    "annotation_id",
    "image_id",
    "area",
    "lesion_width",
    "lesion_height",
    "aspect_ratio",
    "bbox_area"
).show(10)

lesion_features_df = (
    lesion_features_df.join(
        images_df.select(
            col("id").alias("img_id"),
            col("width").alias("image_width"),
            col("height").alias("image_height")
        ),
        lesion_features_df.image_id == col("img_id"),
        "left"
    )
    .drop("img_id")
)

lesion_features_df = lesion_features_df.withColumn(
    "relative_area",
    col("area") / (col("image_width") * col("image_height"))
)

lesion_features_df.select(
    "image_id",
    "area",
    "bbox_area",
    "lesion_width",
    "lesion_height",
    "aspect_ratio",
    "relative_area"
).show(10)

# ============================================================
# ESTADÍSTICAS DE LAS LESIONES
# ============================================================

print("=" * 60)
print("ESTADÍSTICAS DE LAS LESIONES")
print("=" * 60)

lesion_features_df.select(
    "area",
    "bbox_area",
    "lesion_width",
    "lesion_height",
    "aspect_ratio",
    "relative_area"
).describe().show()

lesion_features_df.select(
    "area",
    "lesion_width",
    "lesion_height",
    "aspect_ratio"
).summary(
    "count",
    "mean",
    "50%",
    "min",
    "max"
).show()

# ============================================================
# EXPORTAR A PARQUET PARA LEER DIRECTAMENTE SIN VOLVER A EJECUTAR
# ============================================================

print("=" * 60)
print("EXPORTANDO FEATURES A PARQUET")
print("=" * 60)

OUTPUT_DIR = Path.home() / "arcade-spark-data" / "stenosis_features"

lesion_features_df.write \
    .mode("overwrite") \
    .parquet(str(OUTPUT_DIR))

print(f"✓ Features guardadas en:")
print(OUTPUT_DIR)

# ============================================================
# 9. CERRAR SPARK
# ============================================================

spark.stop()

print("\n" + "=" * 60)
print("SparkSession cerrada correctamente.")
print("=" * 60)
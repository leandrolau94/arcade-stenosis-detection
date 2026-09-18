# ARCADE Stenosis Detection

Automatic coronary artery stenosis detection and localization from X-ray coronary angiography images using deep learning and data engineering.

---

## Overview

This project explores the automatic detection and localization of coronary artery stenosis from X-ray coronary angiography images using a combination of deep learning and data engineering techniques.

The project is based on the ARCADE dataset and focuses on the segmentation of stenosis regions using a U-Net architecture implemented with PyTorch.

In parallel, Apache Spark and PySpark are used for dataset exploration, feature engineering, SQL-based analysis, aggregation, and Parquet data processing.

The project is being developed as an end-to-end machine learning and data engineering workflow, from raw annotated data to model training, evaluation, and visualization.

---

## Project Objective

The main objective is to develop a computer vision pipeline capable of identifying regions corresponding to coronary artery stenosis in angiography images.

The intended workflow is:

1. Analyse the ARCADE dataset.
2. Process the original COCO-style annotations.
3. Generate binary segmentation masks for stenosis regions.
4. Build a PyTorch dataset pipeline.
5. Train a U-Net segmentation model.
6. Evaluate the model using segmentation metrics.
7. Analyse the dataset using PySpark and Spark SQL.
8. Generate engineered features describing stenosis lesions.
9. Store analytical datasets in Apache Parquet format.
10. Visualize predicted stenosis regions on angiography images.
11. Integrate model evaluation and data engineering into a reproducible workflow.

---

## Dataset

The project uses the **ARCADE dataset**, focusing on the stenosis segmentation task.

The current training split contains:

- **1,000 images**
- **1,625 stenosis annotations**
- **997 images containing at least one stenosis**
- **3 images without stenosis**

The annotations are provided in COCO-style JSON format containing:

- `images`
- `annotations`
- `categories`

The stenosis category used in the project is:

```text
category_id = 26
```

The original ARCADE dataset is **not included in this repository**.

---

## Dataset Analysis

The dataset was initially analysed to understand the distribution of stenosis lesions.

The distribution of stenoses per image in the training split is:

| Number of stenoses | Number of images |
|--------------------:|-----------------:|
| 0 | 3 |
| 1 | 583 |
| 2 | 279 |
| 3 | 84 |
| 4 | 32 |
| 5 | 12 |
| 6 | 5 |
| 7 | 2 |

This analysis provides information about the class distribution at image level and is also used to understand the difficulty of the segmentation problem.

---

## Data Engineering with PySpark

Apache Spark is used as a dedicated data engineering and analytical component of the project.

The COCO annotation data is loaded and transformed using **PySpark DataFrames**.

The pipeline performs:

- JSON ingestion
- DataFrame creation
- annotation filtering
- grouping and aggregation
- joins between image and annotation information
- feature engineering
- descriptive statistics
- Spark SQL queries
- Parquet export

### Engineered features

For each stenosis lesion, the following features are generated:

- lesion area
- bounding-box width
- bounding-box height
- bounding-box area
- aspect ratio
- relative lesion area
- image width
- image height
- image identifier
- annotation identifier

For example:

```text
aspect_ratio = lesion_width / lesion_height
```

and:

```text
relative_area = lesion_area / image_area
```

### Spark SQL

Spark SQL is used to analyse the processed data at image level.

The project currently generates statistics such as:

```text
image_id
num_stenoses
total_area
avg_area
max_area
avg_width
avg_height
avg_aspect_ratio
avg_relative_area
```

This produces an analytical dataset with one row per image containing stenosis-related statistics.

### Parquet

The processed Spark datasets are exported to **Apache Parquet**.

Current analytical datasets include:

```text
stenosis_features
image_statistics
```

The Spark environment is currently executed locally using PySpark under Ubuntu/WSL.

---

## Deep Learning

The segmentation model is based on a **U-Net architecture implemented with PyTorch**.

The current pipeline uses:

- grayscale angiography images
- binary stenosis masks
- 256 × 256 input resolution
- batch size of 2
- CPU-based training

The implemented U-Net contains approximately:

```text
1,927,841 trainable parameters
```

The model receives an angiography image and predicts a binary segmentation mask representing the regions associated with stenosis.

---

## Image and Mask Preparation

The original annotations are converted into binary segmentation masks.

The resulting training samples contain:

```text
Input:
    1 × 256 × 256

Target:
    1 × 256 × 256
```

The masks contain binary values:

```text
0 = background
1 = stenosis
```

This allows the problem to be treated as a binary semantic segmentation task.

---

## Training

Several training stages have been performed during the development of the model.

### Initial U-Net training

The first training stage established the initial segmentation model and training pipeline.

The model was subsequently continued from a previously saved checkpoint.

### Continued training

A second training stage used a lower learning rate and continued training from the previous checkpoint.

The best validation results from this stage were:

```text
Validation Dice:      0.4404
Validation IoU:       0.3059
Validation Precision: 0.4595
Validation Recall:    0.5253
```

### Tversky-based training

A third training stage investigated the use of a Tversky-based objective to modify the balance between false positives and false negatives.

The configuration included:

```text
Learning rate: 0.0001
Tversky alpha: 0.6
Tversky beta:  0.4
```

The best validation checkpoint was obtained at epoch 16.

Results:

```text
Validation Dice:      0.4537
Validation IoU:       0.3161
Validation Precision: 0.4791
Validation Recall:    0.5374
```

The best checkpoint was retained rather than automatically using the final training epoch.

These results are experimental and should not be interpreted as clinical performance.

---

## Evaluation Metrics

The current segmentation pipeline evaluates the model using:

### Dice coefficient

Measures the overlap between the predicted segmentation and the ground-truth segmentation.

### Intersection over Union

Measures the intersection between prediction and ground truth relative to their union.

### Precision

Measures the proportion of predicted positive pixels that correspond to the target class.

### Recall

Measures the proportion of ground-truth positive pixels detected by the model.

The project will later investigate these metrics in greater detail across different lesion characteristics.

---

## Visualization

The project includes visualization scripts for inspecting segmentation results.

The objective is to visualize possible stenosis regions directly on the angiography images.

The visualization pipeline can be used to inspect:

- original angiography images
- ground-truth masks
- predicted masks
- detected stenosis regions

This is useful for qualitative analysis of model behaviour and error analysis.

---

## Project Architecture

The current architecture combines data engineering and deep learning:

```text
                         ARCADE DATASET
                               │
                               ▼
                       COCO JSON ANNOTATIONS
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
          PySpark Pipeline               PyTorch Pipeline
                │                             │
                ▼                             ▼
        DataFrames / Spark SQL            Image Dataset
                │                             │
                ▼                             ▼
        Feature Engineering                U-Net
                │                             │
                ▼                             ▼
             Parquet                     Predictions
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                      Model Evaluation
                               │
                               ▼
                         Visualization
```

The two pipelines have different responsibilities:

- **PySpark** handles data processing and analytical workloads.
- **PyTorch** handles model training and inference.

Spark is therefore not artificially inserted into the neural-network training process. It is used where distributed data processing and analytical transformations are appropriate.

---

## Project Structure

The current project structure is:

```text
arcade-stenosis-detection/
│
├── analyze_arcade.py
├── arcade_pytorch_dataset.py
├── continue_training.py
├── first_unet_arcade.py
├── link-descarga-img-ARCADE.txt
├── prepare_masks_arcade.py
├── read_parquet.py
├── spark_eda.py
├── spark_sql.py
├── train_unet_arcade.py
├── train_unet_tversky.py
├── visualize_stenosis.py
├── visualizer.py
│
├── data/
│   └── processed/
│
├── .gitignore
└── README.md
```

The following directories and generated artifacts are intentionally excluded from version control:

```text
arcade/
venv/
entrenamiento_unet/
entrenamiento_continuado/
entrenamiento_tversky/
mascaras_visualizacion/
resultados/
resultados_visualizer/
data/processed/
```

This keeps the Git repository focused on source code and project documentation rather than datasets, model checkpoints, generated images, and other large artifacts.

---

## Technologies

### Machine Learning

- Python
- PyTorch
- U-Net
- Computer Vision
- Image Segmentation
- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Tversky loss

### Data Engineering

- Apache Spark
- PySpark
- Spark DataFrames
- Spark SQL
- Apache Parquet
- Feature Engineering

### Development

- Git
- GitHub
- WSL 2
- Ubuntu
- Python virtual environments

---

## Installation

The project uses Python for the machine learning pipeline and PySpark for data engineering.

A Python virtual environment is recommended for the project.

### Python environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

> A `requirements.txt` file will be added as part of the project reproducibility work.

### PySpark environment

The Spark pipeline is currently executed under Ubuntu/WSL.

PySpark is installed in a dedicated Python virtual environment.

Example:

```bash
python3 -m venv pyspark-venv
source ~/pyspark-venv/bin/activate
pip install pyspark
```

The current Spark version used during development is:

```text
Apache Spark 4.2.0
PySpark 4.2.0
```

---

## Usage

### Dataset analysis

Analyse the ARCADE dataset using:

```bash
python analyze_arcade.py
```

### Mask preparation

Generate binary stenosis masks using:

```bash
python prepare_masks_arcade.py
```

### PySpark EDA

Run the PySpark exploratory data analysis:

```bash
python spark_eda.py
```

### Spark SQL

Run the Spark SQL analytical pipeline:

```bash
python spark_sql.py
```

### U-Net training

Train the initial U-Net model using:

```bash
python train_unet_arcade.py
```

### Continued training

Continue training from a previously saved checkpoint using:

```bash
python continue_training.py
```

### Tversky-based training

Run the Tversky-based training experiment:

```bash
python train_unet_tversky.py
```

### Visualization

Use the visualization scripts to inspect segmentation predictions and stenosis regions.

---

## Reproducibility

Reproducibility is an important part of the project.

The current development environment uses:

- Python
- PyTorch
- PySpark
- Apache Spark
- Ubuntu/WSL
- Git

The project will progressively introduce additional software engineering practices to make experiments easier to reproduce and maintain.

Planned improvements include:

- dependency management
- automated tests
- code formatting
- static analysis
- continuous integration
- containerization
- experiment configuration

---

## Current Status

The following components have been implemented:

- [x] ARCADE dataset analysis
- [x] COCO annotation inspection
- [x] Stenosis category identification
- [x] Binary stenosis mask generation
- [x] PyTorch dataset pipeline
- [x] U-Net implementation
- [x] Initial model training
- [x] Continued model training
- [x] Tversky-based training experiment
- [x] Dice evaluation
- [x] IoU evaluation
- [x] Precision evaluation
- [x] Recall evaluation
- [x] Stenosis visualization
- [x] PySpark installation and configuration
- [x] PySpark DataFrame processing
- [x] Spark feature engineering
- [x] Spark SQL analysis
- [x] Parquet export
- [x] Image-level analytical dataset
- [x] Git version control
- [x] GitHub repository

---

## Planned Next Steps

The next development stages will focus on improving the project from both the machine learning and software engineering perspectives.

### Software Engineering

- [ ] Reorganize the project into a professional `src/` structure
- [ ] Add `requirements.txt`
- [ ] Add automated tests with `pytest`
- [ ] Add code quality checks with Ruff
- [ ] Improve code documentation
- [ ] Introduce type hints
- [ ] Improve configuration management

### Git and CI/CD

- [ ] Establish a consistent Git workflow
- [ ] Improve commit conventions
- [ ] Create GitHub Issues for project tasks
- [ ] Add GitHub Actions
- [ ] Automate tests and code quality checks
- [ ] Build a continuous integration pipeline

### Docker

- [ ] Create a Dockerfile
- [ ] Build a reproducible project environment
- [ ] Containerize the data processing pipeline
- [ ] Integrate Docker into CI

### Machine Learning

- [ ] Improve segmentation performance
- [ ] Investigate additional loss functions
- [ ] Analyse false positives and false negatives
- [ ] Improve model evaluation
- [ ] Evaluate performance across lesion characteristics
- [ ] Investigate image-level and lesion-level metrics
- [ ] Improve qualitative visualization
- [ ] Analyse model predictions using Spark

### Data Engineering

- [ ] Extend the Spark pipeline to model predictions
- [ ] Combine ground-truth and prediction information
- [ ] Perform distributed error analysis
- [ ] Generate additional analytical datasets
- [ ] Improve the data pipeline structure

---

## Long-Term Pipeline

The intended final workflow is:

```text
                    ARCADE DATASET
                          │
                          ▼
                  Data Ingestion
                          │
                          ▼
                 PySpark / Spark SQL
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
                       Parquet
                          │
                          ▼
                    PyTorch Dataset
                          │
                          ▼
                       U-Net
                          │
                          ▼
                     Predictions
                          │
                          ▼
              Spark-based Error Analysis
                          │
                          ▼
                Evaluation & Visualization
                          │
                          ▼
                    CI / Docker
```

The goal is to build an end-to-end project that combines **computer vision, deep learning, data engineering, and software engineering practices**.

---

## Disclaimer

This project is an experimental and educational machine learning project.

The models, metrics, visualizations, and results presented in this repository are not intended for clinical diagnosis, treatment decisions, or direct medical use.

The project is intended for research, learning, and software engineering practice.
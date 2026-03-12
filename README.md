# Industrial Predictive Maintenance Engine (PySpark AFT Survival)

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Apache Spark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-AFT_Survival_Regression-0194E2?style=for-the-badge)

## Executive Summary

This project implements an end-to-end Machine Learning pipeline on **Databricks** to predict the **Remaining Useful Life (RUL)** of industrial machinery (e.g., power supplies) using synthetic IoT sensor data.

By leveraging PySpark's **Accelerated Failure Time (AFT) Survival Regression**, this solution moves beyond simple binary classification ("Will the machine fail?") to answer the critical operational question: *"Exactly how many days until the machine fails?"* This enables the business to:

* **Optimize Maintenance Schedules:** Shift from reactive repairs to predictive, just-in-time maintenance.

* **Reduce Unplanned Downtime:** Identify high-risk machines before catastrophic failure interrupts production.

* **Quantify Sensor Impact:** Mathematically prove exactly how much temperature, vibration, and voltage spikes shrink the expected lifespan of an asset.

## The Tech Stack

* **Core Logic:** Python, PySpark SQL

* **Feature Engineering:** `VectorAssembler`, Statistical Distributions (Normal, Poisson, Uniform)

* **Modeling & Evaluation:** PySpark MLlib (`AFTSurvivalRegression`)

* **Cloud Infrastructure:** Databricks, Unity Catalog (Delta Tables)

## Key Results & Business Impact

The AFT model successfully analyzed 100,000 synthetic IoT records and accurately predicted the remaining days to failure. Furthermore, the model extracted the core coefficients to provide actionable business intelligence:

### Extracted Business Intelligence (Lifespan Impact)

Unlike traditional models, the AFT coefficients directly correlate to the expansion or contraction of the machine's timeline:

* **Voltage Spikes (-0.1000):** Acts as the heaviest risk accelerator. Frequent electrical spikes severely shrink the lifespan.

* **Vibration (-0.0498) & Temperature (-0.0200):** Confirmed as secondary failure accelerators.

* **Maintenance Frequency (+0.0050):** The sole protective factor. The model mathematically proved that frequent maintenance extends the lifespan and delays failure.

## Solution Architecture

This repository is modularized into a 2-stage PySpark pipeline:

### `01_machine_failure_simulation.py`

Engineered a highly realistic synthetic dataset of 100,000 industrial power supplies. Utilized statistical distributions (`randn` for normal distributions, `abs(randn)` for Poisson-like count data) to generate temperature, vibration, and voltage spike features. Calculated a hidden "stress score" resulting in an exponential decay curve to simulate time-to-failure, saving the output to a **Unity Catalog Delta Table**.

### `02_aft_survival_modeling.py`

Ingested the Unity Catalog table and utilized PySpark's `VectorAssembler` to prepare the feature space. Implemented an 80/20 Train/Test split and trained an `AFTSurvivalRegression` model. Extracted lifespan impact coefficients for executive reporting, batch-scored the testing set to predict the exact `predicted_rul` (Remaining Useful Life), and exported the final predictions back to Unity Catalog for dashboard integration.

## How to Run This Project

1. Clone this repository into your Databricks Workspace.

2. Attach the scripts to an active Databricks Compute Cluster.

3. Ensure you have access to `workspace.default` in Unity Catalog (or modify the `output_table` variables to match your catalog schema).

4. Run the scripts sequentially (`01` through `02`).
# Industrial Digital Twin: Turbofan Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-green)

## Project Overview
This project is an end-to-end **Physics-Informed Data Pipeline** and **Interactive Web Dashboard** designed to simulate industrial predictive maintenance. Ingesting actual multi-channel time-series telemetry from the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, this system forecasts the Remaining Useful Life (RUL) of high-stress mechanical assets before catastrophic failure occurs.

Instead of relying on reactive maintenance, this digital twin empowers shop-floor operators with live analytics, dynamic visual alerts, and model explainability to prevent unplanned factory downtime.

## Key Enterprise Features

* **Real-Time Degradation Engine:** Evaluates multi-variate sensor drift (Core Temperature, Static Pressure, Fan Speed) through a pre-trained **Random Forest Regressor** calibrated to exact NASA mechanical thresholds.
* **Interactive Trajectory Visualization:** Dynamically renders historical degradation charts using Pandas and Streamlit, proving how internal mechanical fatigue accelerates over operational cycles.
* **Model Explainability (XAI):** Features a built-in Machine Learning insight module that extracts model weights to show exactly *which* physical sensors (e.g., thermal fatigue vs. mechanical friction) are driving the failure prediction.
* **Data Engineering & Export:** Includes an expandable live view of the ingested Pandas DataFrame with a native one-click **CSV Export pipeline** for maintenance ticketing integration.
* **Dynamic Control Room UI:** Features a custom-injected CSS DOM override that forces the entire dashboard into a critical visual alert state when RUL drops below safety minimums.

## System Architecture

```text
+-----------------------------------+     +-----------------------------------+
|  NASA C-MAPSS Telemetry Logs      |     |  Data Engineering Layer           |
|  (train_FD001.txt)                | --->|  (Pandas, NumPy, Target Gen)      |
+-----------------------------------+     +-----------------------------------+
                                                           |
+-----------------------------------+     +-----------------------------------+
|  Streamlit Control Dashboard      |     |  Predictive Analytics Engine      |
|  (Live UI, Trajectories, CSV)     | <---|  (Scikit-Learn Random Forest)     |
+-----------------------------------+     +-----------------------------------+
```

## Tech Stack

- Core Language: Python
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-Learn, Joblib
- Frontend/UI: Streamlit, Custom HTML/CSS

## Quick Start Guide

1. Clone the repository:

```Bash
git clone https://github.com/flabingo/cmapss1-predictive-maintenance.git
cd cmapss-predictive-maintenance
```

2. Install dependencies:

```Bash
pip install pandas numpy scikit-learn streamlit joblib
```

3. Train the Predictive Model:

Run the backend pipeline to ingest the NASA text logs and serialize the .pkl model.

```Bash
python data_pipeline.py
```

4. Launch the Interactive Dashboard:

```Bash
streamlit run app.py
```

## Engineering Motivation

“Pure computer science models often fail in industrial environments because they treat sensor data as abstract numbers. Traditional mechanical engineering understands the physics but lacks the pipeline to scale it. This project sits explicitly at the intersection: leveraging data science to track true mechanical wear, enabling Industry 4.0 proactive reliability.”
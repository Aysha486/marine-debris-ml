# Marine Debris Detection Using Machine Learning

## 📌 Overview

Marine debris is a growing environmental problem, affecting coastal ecosystems and marine life. This project uses machine learning and data analysis techniques on historical marine debris collection data to analyze debris patterns, estimate future collection levels, identify potential hotspots, and visualize results using an interactive dashboard.

The project was created as **a college group project**.

---

## 🎯 Objectives

- Review historical marine debris collection data.
- Clean and preprocess the country/location-level data.
- Detect patterns and trends within collected marine debris.
- Compare different machine learning algorithms.
- Forecast future debris pickup levels.
- Identify potential geographic hot spots.
- Calculate a Marine Debris Risk Index (MDRI) as defined by the project.
- Provide an interactive dashboard to explore the results.

---

## 📊 Dataset

This project uses data from past marine debris collections. This includes data like:

- Year
- Region/Place
- Amount of people involved
- Pounds of trash picked up
- Distance traveled
- Total items gathered
- Different classes of debris collected

The dataset is cleaned and normalized before being used in analysis and model training.

---

## 🤖 Machine Learning

We examined and compared several machine learning approaches, including:

- Linear Regression
- Random Forest
- Gradient Boosting
- LightGBM
- HistGradientBoosting
- Dummy Mean Baseline

The model performance is evaluated by:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R Squared Score

The project also involves experiments with deep-learning-based forecasting approaches such as:

- LSTM
- GRU
- TCN
- Transformer

---

## 🔮 Forecasting

The forecasting pipeline estimates future debris collection using historical information and engineered time-series features.

Features covered include:

- Previous year values
- Lagged features
- Rolling statistics
- Changes from year to year
- Past trends

The forecasting system can produce future estimates for different countries/locations.

---

## 🌊 Hotspot Analysis

The project also includes geographical analysis to explore areas of higher levels of collected marine debris.

Techniques investigated include:

- DBSCAN
- HDBSCAN
- Visualization by KDE
- K-Means-related analysis
- Analysis of time-based hotspots

Interactive geographical visualizations are created to facilitate exploration of spatial patterns.

> **Note:** There are some geographical representations that use the country/capital centroid coordinates instead of real collection GPS coordinates. As such, the hotspots are intended to be viewed as analytical representations and not the real-world GPS hotspot measures.

---

## 📈 Marine Debris Risk Index (MDRI)

There is a **Marine Debris Risk Index (MDRI)** that was developed specifically for this project.

It consists of normalized measures, which include:

- Total items collected
- Items per individual
- Number of individuals

The project uses a set of weighted components to generate the risk value comparison.

> **Note:** MDRI is a project-specific analytical index and not an officially recognized environmental index.

---

## 🖥️ Interactive Dashboard

The following elements make up the interactive dashboard to explore:

- Historical debris data
- Model comparisons
- Future forecast
- Country/location insights
- Marine debris risk index
- Hotspot maps

Technologies utilized for visualization include:

- Streamlit
- Plotly
- Folium

---

## 🏗️ Project Architecture

```text
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Country/Location Normalization
     ↓
Feature Engineering
     ↓
Train/Test Data
     ↓
Machine Learning Algorithms
     ↓
Model Evaluation
     ↓
Forecasting
     ↓
┌───────────────┬────────────────┐
↓               ↓                ↓
MDRI          Hotspots       Visualization
↓               ↓                ↓
└───────────────┴────────────────┘
                ↓
       Interactive Dashboard

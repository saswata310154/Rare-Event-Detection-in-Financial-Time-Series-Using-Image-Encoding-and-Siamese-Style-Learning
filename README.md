# Rare Event Detection in Financial Time Series using Siamese-Style Neural Networks

This repository presents a novel deep learning framework for **detecting rare and extreme events in financial time series** by transforming temporal price data into images and applying **Siamese-style similarity learning**.  
The approach is designed to handle **severe class imbalance**, **limited labeled data**, and **complex temporal dependencies**, which are common challenges in financial markets.

---

## 📌 Project Overview

Rare financial events such as sudden price drops, crashes, and volatility spikes occur infrequently but have disproportionate impact on portfolios and risk management systems. Traditional time-series models often fail to detect these events reliably.

This project proposes a hybrid framework that combines:
- **Time-series to image encoding**
- **Transfer learning with pre-trained CNNs**
- **Similarity-based (Siamese-style) classification**

The model is evaluated on historical stock market data and demonstrates strong performance under extreme class imbalance.

---

## 🧠 Key Concepts Used

- Rare Event Detection  
- Time-Series Image Encoding  
- Sliding Window Analysis  
- Transfer Learning  
- Convolutional Neural Networks (CNNs)  
- Siamese Neural Networks  
- Distance-Based Classification  
- Financial Time-Series Analysis  

---

## 🏗️ Methodology

### 1. Data Preparation
- Financial OHLC (High, Low, Close) data is collected.
- Rare events are defined using **percentile-based thresholds** on future price changes (e.g., 95th percentile).

### 2. Sliding Window Construction
- Each sample consists of a **25-day historical window**.
- Window size captures sufficient temporal context while remaining computationally efficient.

### 3. Image Encoding (Snake Pattern)
- Numerical time-series windows are normalized and reshaped.
- A **snake-pattern traversal** preserves temporal locality.
- OHLC values are mapped to **RGB channels**.
- Images are resized to `224 × 224` pixels.

### 4. Feature Extraction
- A **pre-trained VGG16 CNN** (ImageNet weights) is used.
- Extracts a **4096-dimensional feature vector** for each window.
- No CNN fine-tuning required → reduces overfitting risk.

### 5. Siamese-Style Similarity Learning
- Manhattan (L1) distance is computed between:
  - Event–Event feature pairs
  - Event–Non-Event feature pairs
- Bootstrap resampling estimates distance distributions.
- Classification is performed via **maximum likelihood estimation**.

---

## 📊 Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

A **time-based train–test split** is used to prevent data leakage.

---

## 📈 Results Summary

- Robust detection under **1%, 5%, and 10% event frequency**
- Clear separation between event and non-event similarity distributions
- Effective handling of extreme class imbalance
- Interpretable similarity-based decision making

> A naive classifier achieves high accuracy but fails to detect rare events.  
> This framework significantly improves rare-event sensitivity while maintaining reliability.

---

## 📂 Repository Structure

├── data/ # Raw and processed financial data
├── notebooks/ # Jupyter notebooks (experiments & analysis)
├── src/ # Core implementation scripts
│ ├── image_encoding.py
│ ├── feature_extraction.py
│ ├── similarity_model.py
│ └── evaluation.py
├── results/ # Metrics, plots, and outputs
├── figures/ # Reconstructed images and visualizations
├── README.md # Project documentation
└── requirements.txt # Dependencies

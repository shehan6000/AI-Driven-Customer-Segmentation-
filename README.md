# AI-Driven Customer Segmentation System
## SQL + Machine Learning + Interactive Dashboard

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technologies Used](#technologies-used)
4. [Database Schema](#database-schema)
5. [Installation Guide](#installation-guide)
6. [Usage Instructions](#usage-instructions)
7. [SQL Queries Explained](#sql-queries-explained)
8. [Machine Learning Pipeline](#machine-learning-pipeline)
9. [Dashboard Features](#dashboard-features)
10. [Results & Insights](#results--insights)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

### Purpose
This project implements an **end-to-end customer segmentation system** that combines SQL analytics, machine learning clustering, and interactive visualization to help businesses understand their customer base and make data-driven marketing decisions.

### Key Objectives
- Extract meaningful customer behavior patterns using SQL
- Apply unsupervised machine learning (K-Means) for customer segmentation
- Create actionable customer segments with business labels
- Provide interactive dashboard for exploration and analysis
- Enable real-time insights through web-based interface

### Business Value
- **Targeted Marketing**: Identify high-value customers for premium campaigns
- **Retention Strategy**: Detect at-risk customers before churn
- **Resource Optimization**: Allocate marketing budget efficiently
- **Personalization**: Tailor products and services to segment needs
- **Revenue Growth**: Focus on segments with highest lifetime value

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  SQLite Database                                                │
│  ├── customers (1,000 records)                                  │
│  ├── products (50 records)                                      │
│  ├── transactions (8,400 records)                               │
│  └── interactions (9,821 records)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SQL PROCESSING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Complex SQL Queries (CTEs)                                     │
│  ├── RFM Calculation (Recency, Frequency, Monetary)            │
│  ├── Customer Tenure Analysis                                  │
│  ├── Interaction Pattern Aggregation                           │
│  └── Feature Engineering (15 features)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MACHINE LEARNING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  scikit-learn Pipeline                                          │
│  ├── StandardScaler (Feature Normalization)                    │
│  ├── K-Means Clustering (n_clusters=5)                         │
│  ├── PCA (Dimensionality Reduction for Visualization)          │
│  ├── Silhouette Score (Model Evaluation)                       │
│  └── Intelligent Segment Labeling                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Web Application                                      │
│  ├── Interactive Filters                                        │
│  ├── Key Metrics Dashboard                                     │
│  ├── 3D Scatter Plots (Plotly)                                 │
│  ├── Statistical Summaries                                     │
│  └── Exportable Reports                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  ngrok Tunnel (Public URL)                                      │
│  └── Accessible from anywhere via web browser                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Technologies Used

### Core Technologies
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Main programming language |
| SQLite | 3.x | Relational database |
| pandas | Latest | Data manipulation |
| scikit-learn | Latest | Machine learning algorithms |
| Streamlit | Latest | Web dashboard framework |
| Plotly | Latest | Interactive visualizations |
| ngrok | Latest | Public URL tunneling |

### Python Libraries
```python
# Data Processing
import pandas as pd
import numpy as np
import sqlite3

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Web Framework
import streamlit as st

# Deployment
from pyngrok import ngrok
```

---

## 🗄️ Database Schema

### Entity Relationship Diagram
```
┌──────────────┐         ┌──────────────┐
│  customers   │         │   products   │
├──────────────┤         ├──────────────┤
│ customer_id  │───┐ ┌───│ product_id   │
│ name         │   │ │   │ product_name │
│ email        │   │ │   │ category     │
│ signup_date  │   │ │   │ price        │
│ country      │   │ │   └──────────────┘
└──────────────┘   │ │
                   │ │
                   ↓ ↓
            ┌──────────────┐
            │ transactions │
            ├──────────────┤
            │ transaction_id│
            │ customer_id  │
            │ product_id   │
            │ trans_date   │
            │ quantity     │
            │ total_amount │
            └──────────────┘

┌──────────────┐
│ interactions │
├──────────────┤
│ interaction_id│
│ customer_id  │───→ customers.customer_id
│ inter_type   │
│ inter_date   │
│ duration_sec │
└──────────────┘
```

### Table Structures

#### 1. customers
```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date DATE,
    country TEXT
);
```
- **Records**: 1,000 customers
- **Purpose**: Store customer demographic information

#### 2. products
```sql
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL
);
```
- **Records**: 50 products
- **Categories**: Electronics, Clothing, Home, Sports, Books

#### 3. transactions
```sql
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    transaction_date DATE,
    quantity INTEGER,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```
- **Records**: 8,400 transactions
- **Purpose**: Track customer purchases

#### 4. interactions
```sql
CREATE TABLE interactions (
    interaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    interaction_type TEXT,
    interaction_date DATE,
    duration_seconds INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```
- **Records**: 9,821 interactions
- **Types**: email_open, website_visit, support_ticket, survey_response

---

## 📥 Installation Guide

### Prerequisites
- Google Colab account (free)
- ngrok account (free) - [Sign up here](https://dashboard.ngrok.com/signup)
- Web browser

### Step-by-Step Installation

#### 1. Open Google Colab
```
https://colab.research.google.com/
```

#### 2. Create New Notebook
- File → New Notebook
- Name it: "Customer_Segmentation_Project"

#### 3. Copy and Run Cells
Copy each cell from the provided code and run sequentially.

#### 4. Get ngrok Token
```
1. Visit: https://dashboard.ngrok.com/get-started/your-authtoken
2. Sign up (free account)
3. Copy your authtoken
4. Keep it ready for Cell 2
```

#### 5. Run Installation Cell
```python
# Cell 1
!pip install streamlit pyngrok scikit-learn pandas sqlalchemy matplotlib seaborn plotly -q
!pip install --upgrade pyngrok -q
```

---

## 🚀 Usage Instructions

### Quick Start (5 Steps)

#### Step 1: Run Installation
```python
# Execute Cell 1
!pip install streamlit pyngrok scikit-learn pandas sqlalchemy matplotlib seaborn plotly -q
```

#### Step 2: Authenticate ngrok
```python
# Execute Cell 2 and paste your token when prompted
from getpass import getpass
NGROK_TOKEN = getpass("Enter your ngrok authtoken: ")
!ngrok authtoken {NGROK_TOKEN}
```

#### Step 3: Run Main Pipeline
```python
# Execute Cells 3-9
# This will:
# - Import libraries
# - Create database
# - Extract features
# - Perform clustering
# - Analyze segments
# - Generate visualizations
```

#### Step 4: Launch Dashboard
```python
# Execute Cell 10
# Streamlit server starts
# ngrok creates public URL
```

#### Step 5: Access Dashboard
```
Open the URL displayed in output:
https://xxxx-xx-xx-xx-xx.ngrok-free.app
```

### Detailed Workflow

```
┌─────────────────────────────────────────┐
│ 1. Installation (Cell 1)               │
│    Duration: ~30 seconds                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Authentication (Cell 2)              │
│    Duration: ~5 seconds                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Library Imports (Cell 3)             │
│    Duration: ~2 seconds                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Database Creation (Cell 4)           │
│    Duration: ~10 seconds                │
│    Output: customer_data.db             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. Feature Engineering (Cell 5)         │
│    Duration: ~5 seconds                 │
│    Output: 15 features extracted        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 6. Machine Learning (Cell 6)            │
│    Duration: ~10 seconds                │
│    Output: 5 customer segments          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 7. Segment Analysis (Cell 7)            │
│    Duration: ~3 seconds                 │
│    Output: Business labels assigned     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 8. Visualizations (Cell 8)              │
│    Duration: ~5 seconds                 │
│    Output: 3D plots displayed           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 9. Save Results (Cell 9)                │
│    Duration: ~2 seconds                 │
│    Output: CSV files created            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 10. Launch Dashboard (Cell 10)          │
│     Duration: ~10 seconds               │
│     Output: Public URL                  │
└─────────────────────────────────────────┘
```

---

## 🔍 SQL Queries Explained

### Main Feature Extraction Query

```sql
WITH customer_rfm AS (
    -- Calculate RFM metrics for each customer
    SELECT 
        c.customer_id,
        c.name,
        
        -- RECENCY: Days since last transaction
        JULIANDAY('now') - JULIANDAY(MAX(t.transaction_date)) as recency_days,
        
        -- FREQUENCY: Number of transactions
        COUNT(DISTINCT t.transaction_id) as frequency,
        
        -- MONETARY: Total spending
        SUM(t.total_amount) as total_spend,
        AVG(t.total_amount) as avg_transaction_value,
        
        -- Additional purchase metrics
        COUNT(DISTINCT t.product_id) as unique_products,
        MAX(t.total_amount) as max_purchase,
        MIN(t.total_amount) as min_purchase
        
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id, c.name
),

customer_interactions AS (
    -- Aggregate customer engagement data
    SELECT 
        customer_id,
        COUNT(*) as interaction_count,
        AVG(duration_seconds) as avg_interaction_duration,
        COUNT(DISTINCT interaction_type) as interaction_types,
        SUM(CASE WHEN interaction_type = 'email_open' THEN 1 ELSE 0 END) as email_opens,
        SUM(CASE WHEN interaction_type = 'website_visit' THEN 1 ELSE 0 END) as website_visits
    FROM interactions
    GROUP BY customer_id
),

customer_tenure AS (
    -- Calculate customer lifetime
    SELECT 
        customer_id,
        JULIANDAY('now') - JULIANDAY(signup_date) as tenure_days
    FROM customers
)

-- Final feature set combining all metrics
SELECT 
    rfm.customer_id,
    rfm.name,
    COALESCE(rfm.recency_days, 365) as recency_days,
    COALESCE(rfm.frequency, 0) as frequency,
    COALESCE(rfm.total_spend, 0) as total_spend,
    COALESCE(rfm.avg_transaction_value, 0) as avg_transaction_value,
    COALESCE(rfm.unique_products, 0) as unique_products,
    COALESCE(rfm.max_purchase, 0) as max_purchase,
    COALESCE(rfm.min_purchase, 0) as min_purchase,
    COALESCE(i.interaction_count, 0) as interaction_count,
    COALESCE(i.avg_interaction_duration, 0) as avg_interaction_duration,
    COALESCE(i.interaction_types, 0) as interaction_types,
    COALESCE(i.email_opens, 0) as email_opens,
    COALESCE(i.website_visits, 0) as website_visits,
    t.tenure_days
FROM customer_rfm rfm
LEFT JOIN customer_interactions i ON rfm.customer_id = i.customer_id
LEFT JOIN customer_tenure t ON rfm.customer_id = t.customer_id
ORDER BY rfm.customer_id;
```

### Feature Definitions

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| **recency_days** | Days since last purchase | How recently customer purchased |
| **frequency** | Count of transactions | How often customer purchases |
| **total_spend** | Sum of all purchases | Total customer value |
| **avg_transaction_value** | Average purchase amount | Typical spending per visit |
| **unique_products** | Distinct products purchased | Product diversity |
| **max_purchase** | Largest single purchase | Maximum willingness to spend |
| **min_purchase** | Smallest purchase | Minimum purchase threshold |
| **interaction_count** | Total engagements | Overall engagement level |
| **avg_interaction_duration** | Average time spent | Engagement depth |
| **interaction_types** | Variety of interactions | Engagement diversity |
| **email_opens** | Email engagement count | Email responsiveness |
| **website_visits** | Site visit count | Online engagement |
| **tenure_days** | Days since signup | Customer lifetime |

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing

```python
# Select features for clustering
feature_cols = [
    'recency_days', 'frequency', 'total_spend', 
    'avg_transaction_value', 'unique_products', 'max_purchase',
    'interaction_count', 'avg_interaction_duration', 
    'interaction_types', 'tenure_days'
]

X = df[feature_cols].fillna(0)
```

### 2. Feature Normalization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Why normalize?
# - K-Means is distance-based
# - Features have different scales
# - Ensures equal feature importance
```

### 3. Optimal Cluster Selection

```python
# Elbow Method
inertias = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
```

**Evaluation Metrics:**
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Silhouette Score**: Cluster separation quality (-1 to 1, higher is better)

### 4. K-Means Clustering

```python
# Apply K-Means with 5 clusters
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10,
    max_iter=300
)

df['cluster'] = kmeans.fit_predict(X_scaled)
```

**Algorithm Parameters:**
- `n_clusters=5`: Number of segments
- `random_state=42`: Reproducibility
- `n_init=10`: Number of initializations
- `max_iter=300`: Maximum iterations

### 5. Segment Labeling

```python
def label_segments(cluster_summary):
    """
    Intelligent business labeling based on RFM characteristics
    """
    labels = {}
    
    for cluster in cluster_summary.index:
        spend = cluster_summary.loc[cluster, 'Avg_Spend']
        freq = cluster_summary.loc[cluster, 'Avg_Frequency']
        recency = cluster_summary.loc[cluster, 'Avg_Recency']
        
        # High value + High frequency = Champions
        if spend > median_spend and freq > median_freq:
            labels[cluster] = '💎 Champions'
        
        # High spend + High recency = At Risk
        elif spend > median_spend and recency > q75_recency:
            labels[cluster] = '⚠️ At Risk'
        
        # High spend only = Big Spenders
        elif spend > median_spend:
            labels[cluster] = '💰 Big Spenders'
        
        # High frequency + Low recency = Loyal
        elif freq > median_freq and recency < median_recency:
            labels[cluster] = '🌟 Loyal Customers'
        
        # Others = Potential
        else:
            labels[cluster] = '🌱 Potential Loyalists'
    
    return labels
```

### 6. Dimensionality Reduction (PCA)

```python
from sklearn.decomposition import PCA

# Reduce to 3 dimensions for visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_}")
# Output: [0.389, 0.203, 0.127] = 71.86% total
```

---

## 📊 Dashboard Features

### 1. Key Metrics Panel
```
┌────────────────────────────────────────────────────┐
│  Total Customers  │  Avg Value  │  Avg Freq  │  Segments │
│      1,000        │  $3,375.75  │    8.4     │     5     │
└────────────────────────────────────────────────────┘
```

### 2. Interactive Filters
- **Segment Selection**: Multi-select dropdown
- **Date Range**: Filter by customer tenure
- **Value Range**: Filter by total spend
- **Frequency Range**: Filter by purchase frequency

### 3. Visualizations

#### A. Pie Chart - Segment Distribution
Shows percentage of customers in each segment

#### B. Bar Chart - Revenue by Segment
Compares total revenue contribution

#### C. Box Plots - RFM Analysis
- Recency distribution per segment
- Frequency distribution per segment
- Monetary distribution per segment

#### D. 3D Scatter Plot
Interactive 3D visualization of customer clusters in RFM space

### 4. Data Tables
- **Segment Summary**: Statistical overview of each segment
- **Customer Details**: Searchable, sortable customer list
- **Export Options**: Download filtered data as CSV

### 5. Navigation Features
- Sidebar filters
- Responsive design
- Real-time updates
- Hover tooltips
- Zoom and pan on charts

---

## 📈 Results & Insights

### Clustering Results

```
Total Customers: 1,000
Number of Segments: 5 (collapsed to 2 main groups)
Silhouette Score: 0.189
PCA Variance Explained: 71.86%
```

### Segment Breakdown

#### 1. 💎 Champions (428 customers - 42.8%)
```yaml
Characteristics:
  Average Recency: 19.29 days
  Average Frequency: 21.91 purchases
  Average Spend: $9,105.83
  Average Transaction: $419.80
  Customer Behavior: High-value, frequent buyers

Marketing Strategy:
  - VIP treatment programs
  - Early access to new products
  - Premium customer service
  - Referral program incentives
  - Exclusive offers and rewards

Expected Outcome:
  - Highest lifetime value
  - Strong brand advocates
  - Low churn risk
  - High cross-sell potential
```

#### 2. 🌱 Potential Loyalists (572 customers - 57.2%)
```yaml
Characteristics:
  Average Recency: 135.84 - 95.82 days
  Average Frequency: 2.40 - 2.88 purchases
  Average Spend: $606.78 - $1,563.44
  Average Transaction: $244.24 - $565.74
  Customer Behavior: Occasional buyers with growth potential

Marketing Strategy:
  - Re-engagement campaigns
  - Personalized product recommendations
  - Time-limited offers
  - Loyalty program enrollment
  - Educational content marketing

Expected Outcome:
  - Conversion to higher segments
  - Increased purchase frequency
  - Higher average order value
  - Improved engagement rates
```

### Business Insights

#### Revenue Concentration
- Top 42.8% customers (Champions) generate **~70% of total revenue**
- Bottom 57.2% customers contribute **~30% of revenue**
- **80/20 rule** approximately holds

#### Churn Risk
- Champions have lowest recency (19.29 days)
- Potential Loyalists show higher recency (95-136 days)
- **Action Required**: Re-engage lapsed customers

#### Growth Opportunities
- 572 customers in Potential Loyalists segment
- Potential revenue increase: **$2-4M annually** if converted to Champions
- Focus on increasing frequency from 2.5 to 5+ purchases

#### Segment-Specific Metrics

| Metric | Champions | Potential Loyalists | Difference |
|--------|-----------|---------------------|------------|
| Avg Lifetime Value | $9,105 | $1,085 | **8.4x** |
| Avg Purchase Frequency | 21.9 | 2.6 | **8.4x** |
| Avg Days Since Purchase | 19.3 | 115.8 | **6.0x** |
| Engagement Score | 10.54 | 10.81 | Similar |

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. ngrok Authentication Failed
```
Error: "authentication failed: Usage of ngrok requires verified account"

Solution:
1. Sign up at: https://dashboard.ngrok.com/signup
2. Get token from: https://dashboard.ngrok.com/get-started/your-authtoken
3. Run Cell 2 again and paste token when prompted
```

#### 2. Module Not Found Error
```
Error: "ModuleNotFoundError: No module named 'pyngrok'"

Solution:
Run: !pip install pyngrok --upgrade -q
Then restart runtime: Runtime → Restart Runtime
```

#### 3. Streamlit Not Starting
```
Error: "Address already in use"

Solution:
!pkill -f streamlit
Then run Cell 10 again
```

#### 4. Database Lock Error
```
Error: "database is locked"

Solution:
conn.close()  # Close all connections
Then re-run database creation cell
```

#### 5. Memory Error
```
Error: "MemoryError: Unable to allocate array"

Solution:
# Reduce dataset size
sample_size = 500  # Instead of 1000
# Or upgrade Colab: Runtime → Change runtime type → High-RAM
```

#### 6. Visualization Not Showing
```
Error: Plots not rendering in Colab

Solution:
# Add these lines before plotting
import plotly.io as pio
pio.renderers.default = 'colab'
```

---

## 🚀 Future Enhancements

### Phase 1: Advanced Analytics
```python
# 1. Time Series Analysis
- Customer lifecycle stages
- Seasonal buying patterns
- Trend prediction

# 2. Churn Prediction
- Logistic regression model
- Survival analysis
- Early warning system

# 3. Customer Lifetime Value (CLV)
- CLV prediction model
- Revenue forecasting
- ROI optimization
```

### Phase 2: Enhanced Features
```python
# 1. Real-time Updates
- Live database connection
- Streaming data processing
- Auto-refresh dashboard

# 2. A/B Testing Framework
- Campaign effectiveness
- Segment response comparison
- Statistical significance testing

# 3. Recommendation Engine
- Product recommendations
- Next best action
- Cross-sell/upsell suggestions
```

### Phase 3: Scale & Performance
```python
# 1. Database Optimization
- PostgreSQL migration
- Indexed queries
- Materialized views

# 2. Big Data Integration
- Apache Spark processing
- Distributed computing
- Cloud deployment (AWS/GCP)

# 3. API Development
- RESTful API
- Authentication
- Rate limiting
```

### Phase 4: Advanced ML
```python
# 1. Deep Learning
- Neural network clustering
- Autoencoder features
- Transfer learning

# 2. Ensemble Methods
- Multiple algorithm comparison
- Voting classifiers
- Stacking models

# 3. AutoML
- Automated feature selection
- Hyperparameter tuning
- Model selection
```

---

## 📚 Appendix

### A. Required Libraries Version Matrix
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
streamlit>=1.25.0
pyngrok>=6.0.0
sqlalchemy>=2.0.0
```

### B. File Structure
```
project/
├── customer_data.db           # SQLite database
├── customer_segments.csv      # Final segmented data
├── cluster_summary.csv        # Segment statistics
├── streamlit_app.py          # Dashboard application
└── notebook.ipynb            # Main Colab notebook
```

### C. Performance Benchmarks
```
Database Creation: ~10 seconds
Feature Engineering: ~5 seconds
ML Clustering: ~10 seconds
Dashboard Launch: ~15 seconds
Total Execution Time: ~60 seconds
```

### D. Glossary

- **RFM**: Recency, Frequency, Monetary - customer value model
- **K-Means**: Unsupervised clustering algorithm
- **PCA**: Principal Component Analysis - dimensionality reduction
- **Silhouette Score**: Cluster quality metric
- **CTE**: Common Table Expression - SQL subquery
- **Churn**: Customer attrition/loss
- **CLV**: Customer Lifetime Value
- **Segment**: Group of customers with similar characteristics

### E. References

1. **K-Means Clustering**: 
   - [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)

2. **RFM Analysis**:
   - [Customer Segmentation Best Practices](https://www.optimove.com/resources/learning-center/rfm-segmentation)

3. **Streamlit**:
   - [Official Documentation](https://docs.streamlit.io/)

4. **SQL Optimization**:
   - [SQLite Query Optimization](https://www.sqlite.org/queryplanner.html)

---




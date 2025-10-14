# Maven Music Customer Churn Analysis

## Executive Summary

Maven Music, a music streaming service, has experienced an unusual increase in customer churn over recent months. This data science project aims to investigate the underlying factors contributing to customer attrition and develop predictive models to support retention strategies. The project follows a structured data science methodology encompassing data collection, cleaning, exploratory analysis, and preparation for machine learning implementation.

## Project Objectives

The primary objectives of this initiative are:

1. **Scope Definition**: Establish project boundaries, define key metrics, and identify relevant stakeholders
2. **Data Collection**: Aggregate customer subscription information and music listening behavior data
3. **Data Cleaning**: Identify and remediate data quality issues, handle missing values, and standardize formats
4. **Exploratory Data Analysis**: Uncover patterns, trends, and relationships within the customer dataset
5. **Data Preparation**: Engineer features and prepare datasets suitable for predictive modeling

## Project Scope

### Data Sources

- **Customer Subscription Data**: Account creation dates, subscription tiers, billing information, and account status
- **Listening History**: User engagement metrics, track preferences, listening frequency, and platform usage patterns
- **Churn Indicators**: Customer termination dates, account cancellation reasons, and tenure information

### Key Variables of Interest

- Target Variable: Customer churn status (binary classification)
- Predictors: Subscription duration, listening frequency, playlist diversity, payment method, subscription tier, user engagement metrics
- Temporal Variables: Subscription start date, last activity date, churn date

### Stakeholders

- Marketing Department: Retention strategy development
- Product Team: Feature enhancement prioritization
- Executive Leadership: Business impact assessment

## Methodology

### 1. Data Gathering

The data collection phase involves aggregating customer and listening activity information from Maven Music's data infrastructure:

```python
import pandas as pd
import numpy as np

# Load customer subscription data
customers = pd.read_csv('data/customers.csv')

# Load listening history data
listening_history = pd.read_csv('data/listening_history.csv')

# Initial data inspection
print(customers.head())
print(listening_history.head())
print(customers.info())
```

### 2. Data Cleaning

Data quality assurance procedures include:

- **Missing Value Detection**: Identify and document missing data patterns
- **Outlier Detection**: Identify anomalous values using statistical methods
- **Data Type Validation**: Ensure appropriate data types for each variable
- **Duplicate Removal**: Eliminate redundant records
- **Data Standardization**: Normalize formats and units

```python
# Check for missing values
print(customers.isnull().sum())

# Remove duplicates
customers = customers.drop_duplicates()

# Handle missing values
customers['last_activity'] = pd.to_datetime(customers['last_activity'])
customers = customers.dropna(subset=['subscription_id'])

# Standardize categorical variables
customers['subscription_tier'] = customers['subscription_tier'].str.lower()
```

### 3. Exploratory Data Analysis

The exploration phase employs descriptive statistics and visualization techniques:

- **Univariate Analysis**: Distribution of individual variables
- **Bivariate Analysis**: Relationships between churn and predictor variables
- **Temporal Analysis**: Trends in customer behavior over time
- **Segmentation Analysis**: Customer cohort comparison

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Descriptive statistics
print(customers.describe())

# Churn rate calculation
churn_rate = customers['churn'].value_counts(normalize=True)
print(f"Churn Rate: {churn_rate[1]:.2%}")

# Visualization: Churn by subscription tier
sns.countplot(data=customers, x='subscription_tier', hue='churn')
plt.title('Customer Churn Distribution by Subscription Tier')
plt.show()

# Listening frequency analysis
listening_metrics = listening_history.groupby('customer_id').agg({
    'track_id': 'count',
    'listen_date': 'max'
}).rename(columns={'track_id': 'total_listens', 'listen_date': 'last_listen'})

customers = customers.merge(listening_metrics, on='customer_id', how='left')
```

### 4. Data Preparation for Modeling

Feature engineering and dataset structuring for machine learning applications:

- **Feature Engineering**: Create derived variables capturing meaningful relationships
- **Temporal Feature Creation**: Extract temporal characteristics from date variables
- **Encoding Categorical Variables**: Transform categorical features for model compatibility
- **Feature Scaling**: Normalize numerical features where appropriate
- **Train-Test Splitting**: Partition data for model validation

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Feature engineering
customers['subscription_duration_days'] = (
    pd.to_datetime('today') - pd.to_datetime(customers['subscription_start_date'])
).dt.days

customers['days_since_activity'] = (
    pd.to_datetime('today') - pd.to_datetime(customers['last_activity'])
).dt.days

# Encode categorical variables
le = LabelEncoder()
customers['tier_encoded'] = le.fit_transform(customers['subscription_tier'])
customers['payment_method_encoded'] = le.fit_transform(customers['payment_method'])

# Prepare features and target
X = customers[['subscription_duration_days', 'days_since_activity', 
               'total_listens', 'tier_encoded', 'payment_method_encoded']]
y = customers['churn']

# Handle missing values in features
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

## Project Deliverables

1. **Data Quality Report**: Documentation of data cleaning processes and issues resolved
2. **Exploratory Analysis Report**: Visual and statistical findings from data exploration
3. **Feature Engineering Documentation**: Description of created features and their rationale
4. **Clean Dataset**: Processed data ready for modeling in subsequent phases
5. **Analysis Notebooks**: Jupyter notebooks containing code and interpretative commentary

## Technical Requirements

- **Programming Language**: Python 3.8+
- **Key Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Data Storage**: CSV format
- **Version Control**: Git

## Installation and Setup

```bash
# Clone repository
git clone https://github.com/your-username/maven-music-churn-analysis.git
cd maven-music-churn-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
maven-music-churn-analysis/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   └── listening_history.csv
│   └── processed/
│       └── cleaned_data.csv
├── notebooks/
│   ├── 01_data_gathering.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   └── 04_data_preparation.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── eda_utils.py
│   └── feature_engineering.py
├── requirements.txt
└── README.md
```

## Future Work

Subsequent phases of this project will include:

- Development of predictive models (logistic regression, random forests, gradient boosting)
- Model evaluation and cross-validation
- Feature importance analysis
- Business recommendations based on predictive insights
- Deployment of retention strategies informed by model outputs

## Author

Tahar GUENFOUD  

## License

This project is licensed under the MIT License - see the LICENSE file for details.
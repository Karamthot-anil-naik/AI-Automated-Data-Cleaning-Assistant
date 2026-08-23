# 🤖 AI-Powered Automated Data Cleaning Platform

> **An intelligent data preprocessing platform that automatically detects, analyzes, and cleans messy datasets using AI and machine learning techniques.**

[![Streamlit](https://ai-automated-data-cleaning-platform-c5n4esjdyjafkejjmjrpjt.streamlit.app/)
---

## 📌 Overview

Real-world datasets are rarely clean.

They often contain:

* Missing values
* Duplicate records
* Incorrect data types
* Outliers
* Inconsistent categorical values
* Invalid values
* Unnecessary columns
* Formatting inconsistencies

The **AI-Powered Automated Data Cleaning Platform** automates these preprocessing tasks through an interactive web interface.

Users can upload a CSV or Excel dataset, analyze its quality, select or automatically apply cleaning strategies, and download the cleaned dataset.

### 🎯 Goal

Reduce the amount of repetitive preprocessing work required before performing:

**Data Analysis → Visualization → Machine Learning → AI Modeling**

---

## ✨ Key Features

### 📂 1. Dataset Upload

Upload datasets directly through the Streamlit interface.

Supported formats:

* CSV
* Excel (`.xlsx`)

---

### 🔍 2. Automated Data Profiling

The platform analyzes the uploaded dataset and generates insights such as:

* Number of rows and columns
* Data types
* Missing-value percentage
* Duplicate records
* Unique values
* Numerical columns
* Categorical columns
* Dataset statistics

---

### 🧹 3. Missing Value Detection & Cleaning

Automatically detects missing values and provides intelligent cleaning strategies.

Supported approaches include:

* Mean imputation
* Median imputation
* Mode imputation
* Forward fill
* Backward fill
* Dropping missing rows

The cleaning strategy can be selected based on the column's characteristics.

---

### ♻️ 4. Duplicate Detection

Automatically identifies duplicate records.

Users can:

* View duplicate count
* Remove duplicate rows
* Compare dataset size before and after cleaning

---

### 📊 5. Outlier Detection

Detect potential outliers using statistical techniques such as:

* IQR Method
* Z-Score

The platform helps identify abnormal observations before model training.

---

### 🔠 6. Data Type Detection

Automatically identifies potentially incorrect data types.

For example:

```text
Age
"21"
"25"
"30"
```

can be converted into:

```text
21
25
30
```

This makes datasets more suitable for analysis and machine learning.

---

### 🏷️ 7. Categorical Data Cleaning

Handles inconsistent categorical values such as:

```text
Male
male
M
MALE
```

and helps standardize them into consistent representations.

---

### 📈 8. Before vs After Analysis

The platform provides a comparison of the dataset before and after cleaning.

Example:

```text
                    Before      After
---------------------------------------
Rows                 10,000      9,850
Missing Values          425          0
Duplicates              150          0
Columns                  25         25
```

---


---

### 💾 10. Download Cleaned Dataset

After preprocessing, users can download the cleaned dataset for further analysis or machine learning.

---

# 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │      User Upload     │
                  │    CSV / Excel File  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Dataset Profiling  │
                  │                      │
                  │ • Data Types         │
                  │ • Missing Values     │
                  │ • Duplicates         │
                  │ • Statistics         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Cleaning Engine     │
                  │                      │
                  │ • Missing Values     │
                  │ • Duplicates         │
                  │ • Outliers           │
                  │ • Data Types         │
                  │ • Categories         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ AI Recommendations   │
                  │                      │
                  │ Cleaning Strategies  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Clean Dataset        │
                  │ Before/After Report  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Download Dataset   │
                  └──────────────────────┘
```

---

# 🛠️ Tech Stack

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| **Python**       | Core programming language           |
| **Pandas**       | Data manipulation and preprocessing |
| **NumPy**        | Numerical operations                |
| **Scikit-learn** | Statistical & ML preprocessing      |
| **Streamlit**    | Interactive web application         |
| **Matplotlib**   | Data visualization                  |
| **Seaborn**      | Statistical visualization           |
| **OpenPyXL**     | Excel file processing               |

---

# 📁 Project Structure

```text
AI-Automated-Data-Cleaning/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_dataset.csv
│
├── modules/
│   ├── __init__.py
│   ├── profiling.py
│   ├── missing_values.py
│   ├── duplicates.py
│   ├── outliers.py
│   ├── data_types.py
│   └── recommendations.py
│
├── utils/
│   ├── __init__.py
│   ├── file_handler.py
│   └── cleaning_utils.py
│
└── screenshots/
    ├── dashboard.png
    ├── profiling.png
    └── cleaning_results.png
```

---

# 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Automated-Data-Cleaning.git
```

### 2. Navigate to the project

```bash
cd AI-Automated-Data-Cleaning
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📋 Example Workflow

### Step 1 — Upload Dataset

```text
Upload Dataset
      ↓
customer_data.csv
```

### Step 2 — Analyze Dataset

The system automatically detects:

```text
Rows:              7,043
Columns:              31
Missing Values:      217
Duplicates:           43
Numerical Columns:    15
Categorical Columns: 16
```

### Step 3 — Generate Recommendations

Example:

```text
✓ Remove duplicate records
✓ Fill numerical missing values using median
✓ Fill categorical missing values using mode
✓ Detect potential outliers
✓ Standardize categorical values
```

### Step 4 — Clean Dataset

Apply the selected preprocessing operations.

### Step 5 — Review Results

```text
Original Dataset
        ↓
Cleaning Pipeline
        ↓
Clean Dataset
```

### Step 6 — Download

Download the processed dataset for further analysis or machine learning.

---

# 🧠 Data Cleaning Pipeline

```python
Raw Dataset
     │
     ▼
Data Validation
     │
     ▼
Data Profiling
     │
     ├── Missing Values
     │
     ├── Duplicates
     │
     ├── Data Types
     │
     ├── Outliers
     │
     └── Categorical Inconsistencies
     │
     ▼
AI Cleaning Recommendations
     │
     ▼
Cleaning Operations
     │
     ▼
Validation
     │
     ▼
Clean Dataset
```

---

# 📊 Example Cleaning Operations

### Missing Values

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

### Duplicate Removal

```python
df = df.drop_duplicates()
```

### Categorical Standardization

```python
df["Gender"] = (
    df["Gender"]
    .str.strip()
    .str.lower()
)
```

### IQR Outlier Detection

```python
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]
```
---

# 🎯 Use Cases

This platform can be useful for:

* Data Scientists
* Machine Learning Engineers
* Data Analysts
* Students
* Researchers
* Business Analysts
* ML Engineers

It can significantly reduce repetitive preprocessing work before model development.

---

# 📈 Project Impact

The platform focuses on one of the most time-consuming parts of a typical machine learning workflow:

```text
Raw Data
   ↓
Cleaning
   ↓
Exploration
   ↓
Feature Engineering
   ↓
Model Training
```

By automating the cleaning stage, the project aims to make the transition from **raw data → ML-ready data** faster and more reliable.

---

# 🔐 Data Privacy

Uploaded datasets should be processed locally or within the configured application environment.

Avoid uploading sensitive or confidential datasets unless the deployment environment has appropriate security controls.

---

# 👨‍💻 Author

**Anil Naik**

B.Tech — Computer Science & Engineering

### Areas of Interest

* Data Science
* Machine Learning
* Generative AI
* NLP
* Python
* SQL
* Data Analytics

---

# ⭐ Why This Project?

This project demonstrates practical knowledge of:

```text
Python
   +
Data Preprocessing
   +
Machine Learning
   +
AI
   +
Data Analytics
   +
Streamlit
```

Rather than building only a machine-learning model, the project focuses on solving a **real-world data engineering and machine learning workflow problem**.

---


---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub!

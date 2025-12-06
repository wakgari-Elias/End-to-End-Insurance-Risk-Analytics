
# End-to-End Insurance Risk Analytics & Predictive Modeling - Task 1: EDA

## Overview

This repository contains **Task-1** of the End-to-End Insurance Risk Analytics & Predictive Modeling challenge for **AlphaCare Insurance Solutions (ACIS)**.  
The goal of Task-1 is to perform **Exploratory Data Analysis (EDA)** on the historical car insurance dataset, covering the period from **February 2014 to August 2015**, and to uncover initial insights into **risk, claims, and premiums**.

The analysis helps in understanding:

- Loss ratio by province
- Claims and premiums by vehicle type and postal code
- Temporal trends in premiums and claims
- Outliers in key financial metrics

This forms the foundation for later tasks including hypothesis testing, predictive modeling, and risk-based premium optimization.

---

## Data

The dataset used is `MachineLearningRating_v3.csv` with the following key columns:

- **Policy / Insurance Info:** `UnderwrittenCoverID`, `PolicyID`, `TransactionMonth`
- **Client Info:** `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`
- **Client Location:** `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`
- **Car Info:** `ItemType`, `Mmcode`, `VehicleType`, `RegistrationYear`, `Make`, `Model`, `Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `VehicleIntroDate`, `CustomValueEstimate`, `AlarmImmobiliser`, `TrackingDevice`, `CapitalOutstanding`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `CrossBorder`, `NumberOfVehiclesInFleet`
- **Plan Info:** `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, `CoverType`, `CoverGroup`, `Section`, `Product`, `StatutoryClass`, `StatutoryRiskType`
- **Payment & Claim:** `TotalPremium`, `TotalClaims`

---

## Task-1: Key Steps Completed

1. **Data Loading & Structure**
   - Loaded CSV and inspected data types
   - Converted `TransactionMonth` to datetime
   - Separated numerical vs categorical features

2. **Data Summarization & Variability**
   - Descriptive statistics for numerical columns
   - Standard deviation and coefficient of variation computed

3. **Missing Values**
   - Checked all columns for missing values

4. **Univariate Analysis**
   - Histograms for numerical features
   - Count plots for categorical features (Province, Gender, VehicleType)

5. **Bivariate & Multivariate Analysis**
   - Created `LossRatio = TotalClaims / TotalPremium`
   - Correlation heatmap for numerical columns
   - Scatter plots for TotalPremium vs TotalClaims

6. **Temporal Trends**
   - Monthly total Premium vs Claims trends

7. **Outlier Detection**
   - Boxplots for key numerical variables (TotalClaims)

8. **Creative Insight Visualizations**
   - Average Loss Ratio by Province
   - Top 15 Vehicle Makes by Average Claims
   - Premium vs Claims colored by PostalCode (large dataset handled with external legend)

---

## Libraries Used

```bash
pip install pandas numpy matplotlib seaborn jupyter
````

Optional libraries for future tasks (Task-2 onwards):

```bash
pip install scikit-learn xgboost shap dvc
```

---

## Repository Structure

```
End-to-End-Insurance-Risk-Analytics/
│
├── data/
│   └── MachineLearningRating_v3.csv
│
├── notebooks/
│   └── Task-1_EDA.ipynb
│
├── src/
│   └── [Python scripts extracted from notebook]
│
├── .gitignore
└── README.md
```

---

## Git / Branch Info

* **Branch:** `task-1`
* **Commits:** Includes descriptive commits for each step:

  * Initialized notebook & loaded CSV
  * Data types & datetime conversion
  * Descriptive statistics & variability
  * Univariate analysis
  * Bivariate analysis, temporal trends, outlier detection
  * Added creative insight plots

---

## Insights / Key Takeaways

* **Geographical Risk:** Some provinces exhibit higher loss ratios.
* **Vehicle Risk:** Certain vehicle makes/models have higher average claims.
* **Temporal Trends:** Premiums and claims show seasonality over 18 months.
* **Outliers:** High-claim policies exist, requiring careful treatment for modeling.

These insights set the stage for **Task-2 (Data Version Control)** and **Task-3 (Hypothesis Testing)**.

---

## Author

**[Elias Wakgari]**
Marketing Analytics Engineer | AlphaCare Insurance Solutions (ACIS)
December 2025

```


```


---

# **Task 2: Reproducible Data Pipeline with DVC**

## **Overview**

In regulated industries such as finance and insurance, it is critical to have **reproducible and auditable analyses**.
Task-2 implements **Data Version Control (DVC)** to ensure:

* Datasets are versioned alongside code
* Analysis can be reproduced anytime
* Compliance with auditing and regulatory standards

---

## **Steps Completed**

### **1. Branch and GitHub Integration**

* Created a new branch: `task-2`
* Merged relevant Task-1 changes via Pull Request
* Descriptive commits added for each DVC setup step

---

### **2. DVC Installation**

```bash
pip install dvc
```

---

### **3. DVC Initialization**

```bash
dvc init
```

* This created `.dvc` folder and config files to manage datasets.

---

### **4. Local Remote Storage Configuration**

1. Create a storage folder:

```bash
mkdir ./dvc_storage
```

2. Add it as a DVC remote:

```bash
dvc remote add -d localstorage ./dvc_storage
```

* `-d` sets it as the **default remote** for the project

---

### **5. Adding and Tracking Dataset**

```bash
dvc add data/MachineLearningRating_v3.csv
```

* This generates a `.dvc` file: `MachineLearningRating_v3.csv.dvc`
* Tracks the dataset without storing it directly in Git

---

### **6. Commit Changes**

```bash
git add data/.gitignore MachineLearningRating_v3.csv.dvc
git commit -m "Task-2: Add dataset and DVC tracking files"
```

---

### **7. Push Data to Local Remote**

```bash
dvc push
```

* Dataset is now versioned and stored in `./dvc_storage`
* Any team member can **pull the exact version** later using:

```bash
dvc pull
```

---

## **Benefits Achieved**

* ✅ Reproducible analysis: data + code versioned together
* ✅ Auditable workflow: dataset versions tracked in Git + DVC
* ✅ Storage efficient: large datasets not stored in Git
* ✅ Ready for future CI/CD integration




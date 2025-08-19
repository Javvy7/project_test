# Data Usage Prediction Project 📊

This project was developed as part of the bootcamp.  
The goal is to predict **local data usage complaints (`data_compl_usg_local_m1`)** using telecom customer data.  

---

## 📂 Project Structure
my_ml_project/
│
├── assets/ # Plots and results (EDA, feature importance, etc.)
│ ├── correlation_matrix.png
│ ├── feature_importances.png
│ └── model_performance.png
│
├── train.py # Script to train the model and save it
├── predict.py # Script to load the model and make predictions
├── requirements.txt # List of dependencies
└── README.md # Documentation


---

## 🚀 Workflow
1. **EDA** – basic exploration of the dataset (`data_usage_production.parquet`)  
2. **Feature Engineering** – applied using `Pipeline` and `ColumnTransformer`  
   - Handling categorical & numerical features  
   - Missing values treatment  
   - Scaling & encoding where necessary  
3. **Model Training** – trained a `RandomForestRegressor`  
4. **Model Evaluation** – used metrics such as:
   - RMSE (Root Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - R² Score  
5. **Model Serialization** – model is saved as a `.pkl` file  
6. **Prediction** – new inputs are passed into the trained model via `predict.py`  

---

## 📊 Results
- Best model: **RandomForestRegressor**  
- Evaluation metrics:
  - RMSE: *to be added*
  - MAE: *to be added*
  - R²: *to be added*  

📌 See visualizations in the `assets/` folder.

---

## ⚙️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Train the model
python train.py


This will train the model and save it (e.g., model.pkl).

3️⃣ Make predictions
python predict.py


This script loads the trained model and outputs predictions for sample data.

🛠️ Requirements

Python 3.9+

pandas

numpy

scikit-learn

joblib

matplotlib / seaborn (for visualization)

Install them all with:

pip install -r requirements.txt

📌 Notes

Dataset: data_usage_production.parquet

Target column: data_compl_usg_local_m1

Index column: telephone_number
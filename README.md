# Airbnb Investment Tool

🧠 **Streamlit App for Predicting Airbnb Daily Prices and investment risk in Lisbon**

This project is a simple yet practical application designed to simulate the daily rental price of an Airbnb property located in Lisbon. It was developed as part of the postgraduate program in **Data Science for Business Strategy (NOVA FCT)**.

## 🎯 Project Objective

The app allows potential Airbnb investors to estimate the daily price of a property based on two main user inputs:

- **Neighborhood (Freguesia)** within Lisbon
- **Property Type (Tipologia)** from T0 to T6

The price is predicted using a **Linear Regression model** trained on publicly available data from [Inside Airbnb](https://insideairbnb.com/lisbon/).

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas, Scikit-learn, Joblib**
- **Streamlit** for the user interface
- **Git + GitHub** for version control

---

## 📁 Project Structure

```

airbnb-price-estimator/
├── app/                    # Streamlit frontend
│   └── frontend.py
├── model/                  # Serialized ML model
│   └── modelo\_regressao.pkl
├── data/                   # Raw or processed data
│   └── listings\_lisboa.csv
├── src/                    # Scripts for training and preprocessing
│   └── train\_model.py
├── requirements.txt        # Python dependencies
└── README.md

````

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/airbnb-investment-tool-lisbon.git
cd airbnb-investment-tool-lisbon
````

### 2. Create a Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

```bash
python src/train_model.py
```

### 4. Run the App

```bash
streamlit run app/frontend.py
```

---

## 📦 Model Details

* **Type**: Linear Regression
* **Features**: `neighbourhood_cleansed`, `tipologia`
* **Target**: `price`
* **Preprocessing**: One-Hot Encoding for categorical features

---

## 📊 Data Source

* Dataset: `listings.csv`
* Source: [Inside Airbnb – Lisbon](https://insideairbnb.com/lisbon/)
* Filtered for properties located in the Lisbon municipality

---

## 🧩 Future Enhancements

* Add more predictors (e.g., number of reviews, amenities, room type)
* Use more advanced models (e.g., XGBoost, Neural Networks)
* Deploy publicly (e.g., Streamlit Cloud, Render, Hugging Face Spaces)
* Add dashboards or scenario simulation for investment optimization

---

## 👨‍🎓 Authors

Developed by students of the **Postgraduate Program in Data Science for Business Strategy**
NOVA School of Science and Technology — 2025

```




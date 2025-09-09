# Laptop Price Prediction (Group D)

## Overview
This project predicts the price of laptops using machine learning, based on their features (brand, hardware specs, etc.). It includes data cleaning, exploratory data analysis (EDA), feature engineering, linear regression modeling, evaluation, and a deployment-ready Streamlit web app for real-time predictions.

## Features
- Cleans and preprocesses raw laptop data (handles missing values, outliers, encoding).
- Explores data visually (histograms, scatter plots, box plots, heatmaps).
- Builds and evaluates a linear regression model.
- Deploys a user-friendly web app for price predictions.
- Version-controlled workflow using Git/GitHub.

## Project Structure
- `notebook.ipynb`: Jupyter Notebook with code, explanations, EDA, and modeling.
- `app.py`: Streamlit app source code.
- `datasets/`: Raw and processed datasets.
- `models/`: Saved model files.
- `requirements.txt`: Python dependencies.
- `report.docx`: Project summary and usage guide.

## Running the Streamlit Application

### 1. Clone the repository
```bash
git clone https://github.com/Idighekere/laptop-price-prediction-ml-model
cd laptop-price-prediction-ml-model
```

### 2. Install requirements
Make sure you have Python 3.7+ installed. Install required libraries:
```bash
pip install -r requirements.txt
```
or manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

By default, this will launch the app in your web browser at [http://localhost:8501](http://localhost:8501).

## Usage

- Fill in the required laptop specs in the form.
- Click the **Predict Price** button.
- The model returns the predicted laptop price instantly.


## System Requirements
- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas, numpy, matplotlib, seaborn

## How to Contribute
1. Fork the repo and create your branch (`git checkout -b feature/YourFeature`)
2. Commit changes with clear messages
3. Push to the branch and open a Pull Request

## Team Members
- [Udo, Idighekere Udeme](https://github.com/idighekere)
- [Edet, Unwana Michael](https://github.com/lightnonstop)
- [Ossai, Olivia Chioma](https://github.com/Olive-Ai10)
- [Nlemedim, Richard, Chinwemeri ](https://github.com/richard-nlemedim)
- [Ukpong, Semaediong Francis](https://github.com/Semaediong01)
- [Umoh, Ubongabasi Nyeneime](https://github.com/ubongabasiumoh266-cpu)
- [Akpan Hope Idoroyie]()
- [Umanah Anammbiet Pius]()
- [Sam, Henry Emmanuel]()
- [Urua, Edikan Usen]()

## License
[MIT](LICENSE)

## Acknowledgments
- [Laptop Price Dataset on Kaggle](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)

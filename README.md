# Telecom Churn Prediction Project

This project focuses on predicting customer churn for a telecom company using a rich dataset that includes features such as call failures, customer complaints, subscription length, charge amount, and more. The primary objective is to uncover patterns and relationships within the data that can help understand factors contributing to customer churn.

Through an in-depth Exploratory Data Analysis (EDA), the aim is to gain insights into the distribution of these features for both churned and non-churned users. This analysis will help to identify key differences that may contribute to actionable recommendations for the telecom company to improve customer retention.

The project includes a backend server for model prediction and a frontend for user interaction, making it a comprehensive solution for churn prediction. The model used for prediction is a gradient boosting model trained using the XGBoost library.



## Project Structure

- `backend/`: Contains the backend server code, implemented in Python using FastAPI.
- `frontend/`: Contains the frontend code, implemented using Streamlit.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis, feature engineering, and model training.
- `data/`: Contains the dataset used for model training.
- `outputs/`: Contains the trained model.

## Setup and Installation

1. Clone the repository.
2. Install the required dependencies using the `requirements.txt` file in both the `backend/` and `frontend/` directories.
3. Build the Docker images for both the backend and frontend using the provided Dockerfiles.

## Usage

1. Run the backend server.
2. Start the frontend application.
3. Use the frontend application to input customer data and receive churn predictions.

## Data

The data used in this project is stored in `data/customer_churn.csv`. The data includes features such as call failures, complaints, subscription length, charge amount, tariff plan, age, and customer value.

The dataset used in this project is sourced from the [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

For a comprehensive analysis of the data, please refer to the `eda.ipynb` notebook.

## Model

The model used for prediction is a gradient boosting model trained using the XGBoost library. The model is trained in the `main.py` script and stored in MLflow.

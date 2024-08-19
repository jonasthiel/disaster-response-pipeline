# Disaster Response Pipeline Project

### Project Overview

This project focuses on building a machine learning model to classify messages sent during disasters into one or more of 36 predefined categories. These categories cover critical areas such as Aid Related, Medical Help, Search and Rescue, and others. Accurate classification of these messages ensures they are routed to the appropriate disaster relief agencies, facilitating a more effective and timely response.

The project involves developing an ETL (Extract, Transform, Load) pipeline to process the data and a Machine Learning pipeline to train and optimize the classification model. Since a single message can belong to multiple categories, the model addresses a multi-label classification problem.

The dataset used in this project is provided by Figure Eight (now Appen) and contains real messages sent during disaster events. The final model is integrated into a web application, allowing users to input a message and receive instant classification results. This web app serves as a practical tool for emergency responders to quickly assess and prioritize disaster-related communications.

### Files in the Repository

The repository contains the following directories and files:

- **app/**
  - **template/**
    - **master.html**: The main page of the web application.
    - **go.html**: The page that displays the classification results.
  - **run.py**: The Flask script that runs the web application.

- **data/**
  - **disaster_categories.csv**: The dataset containing disaster category information to be processed.
  - **disaster_messages.csv**: The dataset containing messages to be processed.
  - **process_data.py**: The script to process the data and save it to an SQLite database.
  - **InsertDatabaseName.db**: The SQLite database where the cleaned data is stored.

- **models/**
  - **train_classifier.py**: The script to train the machine learning model.
  - **classifier.pkl**: The saved machine learning model after training.

- **README.md**: The README file containing the project overview, setup instructions, and other relevant information.

### Instructions

Follow these steps to set up and run the project:

1. Set up the database and model (only necessary if the `.db` or `.pkl` files do not already exist):
   - To run the ETL pipeline that cleans the data and stores it in a database, run the following command: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run the machine learning pipeline that trains the classifier and saves the model, use the command: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the web application by executing the command: `python app/run.py`

3. Access the web app by navigating to [http://0.0.0.0:3001/](http://0.0.0.0:3001/) in your web browser.

### Web App Features

- **Message Classification**: Input a disaster-related message and receive classification results across various categories.

- **Data Visualizations**: The app displays visualizations of the disaster data.
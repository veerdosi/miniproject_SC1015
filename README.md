# DSAI Mini Project

SC1015 Mini Project 2024 (Group)

## Title: Early Prediction of Heart Failure Using Clinical Features

### Problem Statement

**Introduction**: Heart disease remains a leading cause of morbidity and mortality worldwide, posing significant public health challenges. Early detection of heart failure, a major form of heart disease, is critical for improving patient outcomes and managing healthcare resources efficiently.

**Problem Statement**: Despite advancements in medical science, many individuals at risk of heart failure remain undiagnosed until the disease progresses to advanced stages. Traditional diagnostic methods can be invasive, expensive, and not universally accessible, creating a need for alternative early detection techniques.

### Objectives

To develop a predictive model that utilizes easily obtainable clinical features to identify individuals at high risk of heart failure, facilitating early intervention and potentially saving lives.

### Dataset Description

The dataset, titled “Heart Failure Prediction Dataset,” comprises medical records from patients, featuring variables such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more, culminating in a binary indicator of heart disease presence. This comprehensive dataset serves as the foundation for our predictive modeling.

### Methodology: Data Visualisation

- **Basic Data Summary**: Prints the shape of the dataset and displays the first 5 rows.
- **Summary statistics for numerical variables**: Displays summary statistics such as count, mean, standard deviation, minimum, maximum, and quartiles for numerical variables.
- **Distribution plots for all numerical variables**: Plots histograms for each numerical variable to visualize their distributions.
- **Count plots for all categorical variables**: Plots count plots for each categorical variable to visualize their distributions.
- **Correlation heatmap**: Displays a heatmap to visualize the correlation between numerical variables.
- **Pairplot to visualize relationships between numerical variables**: Plots a pairplot to visualize relationships between numerical variables.

### Methodology: Data Modelling

## Heart Disease Prediction Code Documentation

### Introduction

This repository contains code for predicting heart disease using machine learning techniques. The code encompasses data preprocessing, model training, evaluation, and generating prediction reports.

### Methodology

#### Data Preprocessing

- **Loading and Cleaning Data**: The heart disease dataset is loaded using pandas, followed by dropping rows with missing values for simplicity.
- **Feature Engineering**: Interaction terms between 'Age' and 'Cholesterol', and 'Age' and 'RestingBP' are added to unveil potential non-linear relationships. Polynomial features for 'Age' and 'MaxHR' are also generated to capture more complexity in the data.
- **Segregating Features**: Features are segregated into categorical and continuous types to tailor the preprocessing steps.
- **Preprocessing Pipeline**: A ColumnTransformer is employed to apply appropriate transformations—scaling for continuous features using StandardScaler and encoding categorical variables via OneHotEncoder.

#### Model Training and Evaluation

- **Models and Hyperparameters**: Sets up a dictionary mapping model names to tuples of model instances and their respective hyperparameter grids for RandomizedSearchCV.
- **Pipeline Creation**: For each model, a pipeline is created that includes preprocessing steps and the classifier itself.
- **Grid Search**: Applies RandomizedSearchCV to the pipeline to find the best hyperparameters based on cross-validation, using accuracy as the scoring metric.
- **Fitting Models**: Trains the model using the best hyperparameters found for each classifier on the training dataset.
- **Evaluation**: Predicts on the test set and calculates evaluation metrics like accuracy, precision, recall, F1 score, and ROC AUC for each model.
- **Stacking**: Combines predictions from Random Forest and Gradient Boosting models using a Logistic Regression meta-model to improve performance.

#### Neural Network Model

- **Preprocess the Data**: Applies transformations to the training and test data for consistency.
- **Define the Model**: Constructs a neural network with input layer sized to the number of features, followed by two hidden layers with ReLU activation, and an output layer with a sigmoid activation function for binary classification.
- **Compile, Train, and Evaluate the Model**: Compiles the model with the Adam optimizer and binary crossentropy loss, trains it on the processed training data, and evaluates its performance on the processed test data.

### Results

- The stacked model exhibited promising performance with an accuracy of 90.22%, high precision (92.38%), robust recall (90.65%), and a balanced F1 score (91.51%).
- The neural network achieved an accuracy of 85.87%, which was lower than the stacked model's accuracy.

### Report Generation

- A PDF report is generated dynamically based on user inputs and model predictions.
- The report includes user information, predicted risk scores, risk level categorization, and visualization graphs.

### Usage

Run the Streamlit app to predict heart failure risk by providing user information such as age, sex, resting blood pressure, cholesterol levels, and resting ECG results.

### Conclusion

The project successfully established a predictive model capable of early heart failure detection based on clinical features. This model can serve as a tool for healthcare professionals to prioritize high-risk patients for further testing and intervention, potentially reducing heart failure incidences and improving patient outcomes.

### Future Work

Further research could explore integrating additional data sources, such as genetic markers or lifestyle factors, to enhance the model's accuracy. Moreover, deploying the model into a real-world clinical setting represents an exciting frontier for transforming heart disease diagnosis and treatment strategies.

### Implementation Notes

Throughout the analysis and model development process, detailed documentation and comments in the code ensure clarity and reproducibility of results. These notes serve as a guide for other researchers or practitioners wishing to apply or extend this work.

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

Here are the updated pointers incorporating the specific details from your provided code snippet, which refines the overall workflow of data preprocessing, model definition, compilation, training, and evaluation:

1. **Preprocess the Data**:

   - Applies transformations to both training and test data using a preprocessor to ensure consistency. This includes fitting the preprocessor to the training data and transforming both training and test datasets.

2. **Define the Model**:

   - Constructs a neural network that starts with an input layer sized to the number of features from the processed data.
   - Includes batch normalization after the input layer to standardize inputs to the next layer, improving stability and speed of training.
   - Adds two hidden layers with 128 and 64 neurons respectively, both featuring ReLU activation and L2 regularization to reduce overfitting. Each layer is followed by dropout layers with dropout rates of 0.3 for the first two and 0.2 for the last hidden layer (32 neurons).
   - Concludes with an output layer using a sigmoid activation function for binary classification, suitable for producing a probability output.

3. **Compile, Train, and Evaluate the Model**:
   - Compiles the model using a variety of optimizers (Adam, SGD with momentum, and RMSprop) and loss functions (binary crossentropy and Binary Focal Crossentropy), tailoring the model to different training dynamics and focus on class imbalance.
   - Implements early stopping during training to halt training when the validation loss ceases to improve, preventing overfitting and enhancing generalization by restoring the best weights observed.
   - Trains the model in silent mode (without verbose output) over 200 epochs with a batch size of 32 and a validation split of 20%, providing a method to monitor and prevent overfitting while optimizing the model parameters.
   - Evaluates the model on the processed test data, calculating key performance metrics such as accuracy, precision, recall, F1 score, and ROC AUC to assess its predictive power.
   - Visualizes training history for accuracy and loss to identify trends and potential issues in learning, such as overfitting or not learning adequately.

These updates encapsulate the procedural details, showcasing a thorough approach to building and evaluating a neural network model that is both robust and tuned for optimal performance on binary classification tasks.

### Results

---

When evaluating the performance of the Stacked Model against the models trained using different optimizers (Adam, SGD, and RMSprop), we analyze their accuracy, precision, recall, F1 score, and ROC AUC to determine the overall best model.

1. **Accuracy**:

   - **Stacked Model:** 90.22%
   - **Adam:** 88.04%
   - **SGD:** 86.41%
   - **RMSprop:** 88.04%
   - The Stacked Model has the highest accuracy.

2. **Precision**:

   - **Stacked Model:** 92.38%
   - **Adam:** 89.72%
   - **SGD:** 89.42%
   - **RMSprop:** 89.72%
   - The Stacked Model outperforms in terms of precision, making the fewest false positive errors.

3. **Recall**:

   - **Stacked Model:** 90.65%
   - **Adam:** 89.72%
   - **SGD:** 86.92%
   - **RMSprop:** 89.72%
   - Again, the Stacked Model leads with the highest recall, effectively identifying most true positives.

4. **F1 Score**:

   - **Stacked Model:** 91.51%
   - **Adam:** 89.72%
   - **SGD:** 88.15%
   - **RMSprop:** 89.72%
   - The Stacked Model exhibits the best balance between precision and recall.

5. **ROC AUC**:
   - **Stacked Model:** 90.13%
   - **Adam:** 93.49%
   - **SGD:** 93.31%
   - **RMSprop:** 93.62%
   - The neural network models outperform the Stacked Model in this metric, with RMSprop achieving the highest ROC AUC.

The **Stacked Model** consistently achieves higher accuracy, precision, recall, and F1 score than the models trained with Adam, SGD, and RMSprop. This indicates a superior balance of identifying true positives and minimizing false positives while generally making more correct predictions across the board.

The only metric where the neural network models (particularly RMSprop) excel is ROC AUC, which measures the model's ability to discriminate between classes at various threshold settings. RMSprop's slightly higher ROC AUC suggests it might be better suited for applications where the discrimination between classes is more critical than the absolute number of correct predictions.

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

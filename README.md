# Predicting Heart Disease: An End-to-End Machine Learning Project ❤️🩺

-----

## 🚀 Project Overview

This project aims to develop a machine learning model capable of **predicting whether or not a patient has heart disease** based on their medical attributes. Leveraging various Python-based machine learning and data science libraries, this notebook demonstrates an end-to-end approach, from problem definition and data exploration to modeling and evaluation. The ultimate goal is to build a reliable predictive tool that could assist in early diagnosis and patient care.

-----

## 1\. Problem Definition

The central question guiding this project is:

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

-----

## 2\. Data

The dataset used in this project originates from the **Cleveland Heart Disease Data** available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease). A version of this dataset is also accessible on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data).

The dataset contains various medical attributes for patients, with the `target` variable indicating the presence (1) or absence (0) of heart disease.

### Initial Data Snapshot (`df.head()`):

| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|
| 63  | 1   | 3  | 145      | 233  | 1   | 0       | 150     | 0     | 2.3     | 0     | 0  | 1    | 1      |
| 37  | 1   | 2  | 130      | 250  | 0   | 1       | 187     | 0     | 3.5     | 0     | 0  | 2    | 1      |
| 41  | 0   | 1  | 130      | 204  | 0   | 0       | 172     | 0     | 1.4     | 2     | 0  | 2    | 1      |
| 56  | 1   | 1  | 120      | 236  | 0   | 1       | 178     | 0     | 0.8     | 2     | 0  | 2    | 1      |
| 57  | 0   | 0  | 120      | 354  | 0   | 1       | 163     | 1     | 0.6     | 2     | 0  | 2    | 1      |

-----

## 3\. Evaluation

The success of this proof-of-concept project is defined by a clear evaluation metric:

> If we can reach **95% accuracy** at predicting whether or not a patient has heart disease, we will pursue the project further.

-----

## 4\. Features

The dataset includes the following features, providing a comprehensive view of patient health:

  * **age**: Age in years
  * **sex**: (1 = male; 0 = female)
  * **cp**: Chest pain type (e.g., typical angina, atypical angina, non-anginal pain, asymptomatic)
  * **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
  * **chol**: Serum cholesterol in mg/dl
  * **fbs**: (Fasting blood sugar \> 120 mg/dl) (1 = true; 0 = false)
  * **restecg**: Resting electrocardiographic results
  * **thalach**: Maximum heart rate achieved
  * **exang**: Exercise induced angina (1 = yes; 0 = no)
  * **oldpeak**: ST depression induced by exercise relative to rest
  * **slope**: The slope of the peak exercise ST segment
  * **ca**: Number of major vessels (0-3) colored by fluoroscopy
  * **thal**: Thalassemia type (1 = normal; 2 = fixed defect; 3 = reversable defect)
  * **target**: The prediction target (1 = heart disease; 0 = no heart disease)

-----

## 5\. Data Exploration (EDA) and Preprocessing

The exploratory data analysis phase focused on understanding the dataset's characteristics and preparing it for modeling.

  * **Data Loading**: The `heart-disease.csv` file was loaded into a pandas DataFrame.
  * **Initial Inspection**:
      * `df.head()` was used to view the first few rows.
      * `df.info()` confirmed that there were **no missing values** across any of the 14 columns and that data types were appropriate (mostly `int64` and `float64`).
      * `df.describe()` provided descriptive statistics, offering insights into the distribution of numerical features.
  * **Target Distribution**:
      * `df["target"].value_counts()` showed the class distribution: 165 instances with heart disease (target=1) and 138 instances without heart disease (target=0). This indicates a relatively balanced dataset.
      * A **bar plot** visualized this distribution, making it easy to see the counts for each target class.
  * **Feature Relationships**:
      * **Heart Disease Frequency according to Sex**: `pd.crosstab(df.target, df.sex)` revealed the relationship between gender and heart disease presence. The accompanying bar plot visually represented this, showing that while more males were in the dataset, a higher proportion of females had heart disease relative to their group size.
      * **Age vs. Max Heart Rate for Heart Disease**: A scatter plot was generated to explore the relationship between `age`, `thalach` (maximum heart rate achieved), and the `target` variable. This visualization helped identify potential patterns and clusters among patients with and without heart disease across different age and heart rate ranges.

-----

## 6\. Modelling

This project utilized a suite of machine learning classification models from `scikit-learn` to predict heart disease.

### Tools and Libraries:

  * **Data Analysis & Manipulation**: `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`
  * **Models**: `LogisticRegression`, `KNeighborsClassifier`, `RandomForestClassifier`
  * **Model Evaluation**: `train_test_split`, `cross_val_score`, `RandomizedSearchCV`, `GridSearchCV`, `confusion_matrix`, `classification_report`, `precision_score`, `recall_score`, `f1_score`, `RocCurveDisplay`

### Model Training & Evaluation Strategy:

1.  **Data Splitting**: The dataset was split into training and testing sets using `train_test_split` to ensure unbiased model evaluation.
2.  **Model Selection**: Three different classification models were considered:
      * **Logistic Regression**: A strong baseline for binary classification.
      * **K-Nearest Neighbors (KNeighborsClassifier)**: A non-parametric, instance-based learning algorithm.
      * **Random Forest Classifier**: An ensemble method known for its high accuracy and robustness.
3.  **Hyperparameter Tuning**: `RandomizedSearchCV` and `GridSearchCV` were employed to systematically find the optimal hyperparameters for each model, maximizing their performance.
4.  **Performance Metrics**: The models were evaluated using:
      * **Accuracy**: Overall correct predictions.
      * **Precision**: Proportion of positive identifications that were actually correct.
      * **Recall**: Proportion of actual positives that were correctly identified.
      * **F1-score**: Harmonic mean of precision and recall.
      * **Confusion Matrix**: A table showing true positives, true negatives, false positives, and false negatives.
      * **ROC Curve and AUC Score**: Measures the ability of a classifier to distinguish between classes.
      * **Feature Importance**: For tree-based models like Random Forest, feature importance was visualized to understand which medical attributes were most influential in the predictions.

### Results (Example from a typical run, not directly from snippet output but common for this dataset):

After training and tuning, the models were evaluated. A representative classification report might look like this (actual results would be in the notebook):

```
              precision    recall  f1-score   support

           0       0.85      0.88      0.86       138  (No Disease)
           1       0.89      0.86      0.87       165  (Heart Disease)

    accuracy                           0.87       303
   macro avg       0.87      0.87      0.87       303
weighted avg       0.87      0.87      0.87       303
```

  * **Accuracy**: The models typically achieve an accuracy around **85-90%**, aiming for the 95% target set in the evaluation phase.
  * **Feature Importance**: Visualizations of feature importance (e.g., a bar plot of `feature_dict`) highlight which medical attributes, such as `cp` (chest pain type) or `thalach` (maximum heart rate achieved), are most predictive of heart disease.

-----

## 7\. Experimentation

The project emphasizes an iterative approach to machine learning. If the evaluation metric (95% accuracy) is not met, the notebook prompts further experimentation:

  * **Data Collection**: Could more data improve model performance?
  * **Model Exploration**: Trying more advanced models like CatBoost or XGBoost.
  * **Model Improvement**: Further refining current models beyond initial tuning.
  * **Deployment**: If the model is sufficiently accurate, how would it be exported and shared for practical use?

-----

## ✨ Conclusion

This project provides a robust framework for predicting heart disease using machine learning. Through detailed data exploration, strategic model selection, and rigorous evaluation, it demonstrates the potential of data science to contribute significantly to healthcare. The insights gained from feature importance can also guide medical research and diagnostic focus.

-----

## 🚀 How to Run

To run and explore this project:

1.  **Download the Notebook**: Obtain the `end-end-heart-disease-classification.ipynb` file.
2.  **Download Data**: Download the `heart-disease.csv` dataset. It can be found on the [Kaggle Heart Disease Dataset page](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data). Place it in the same directory as your notebook.
3.  **Install Dependencies**: Install the necessary Python libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Run the Notebook**: Open `end-end-heart-disease-classification.ipynb` using Jupyter Notebook or JupyterLab and execute the cells sequentially to reproduce the analysis and model training.

-----

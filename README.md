# Machine Learning for U.S. Car Market and Pricing Predictions
To optimize pricing, we propose machine learning models to predict car prices based on various factors. 
This will help us understand market dynamics and make informed decisions about product design, marketing, and pricing.
This project utilizes machine learning to predict car prices in the U.S. market. 
It involves data analysis, preprocessing, feature selection, and model implementation using various regression algorithms.

## Dataset

The dataset used for this project is the "Car Price Prediction" dataset. It contains information about various car features and their corresponding prices.

[Dataset Link](https://github.com/aneeshmurali-n/Machine-Learning-for-U.S.-Car-Market-and-Pricing-Predictions/raw/refs/heads/main/CarPrice_Dataset.csv)

## Methodology

1. **Data Loading and Preprocessing:** Load the dataset, handle missing values, and perform necessary preprocessing steps like outlier handling and data transformation.
2. **Feature Importance Analysis:** Identify significant variables affecting car prices using correlation analysis and SelectKBest feature selection technique.
3. **Model Implementation:** Implement five regression algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regressor.
4. **Model Evaluation:** Evaluate the performance of each model using metrics such as Mean Squared Error (MSE), R-squared (R2), and Mean Absolute Error (MAE).
5. **Hyperparameter Tuning:** Perform hyperparameter tuning using GridSearchCV to optimize the models and improve their performance.

## Results

The Random Forest Regressor emerged as the best-performing model overall, achieving the highest R2 score and lowest errors. 
Hyperparameter tuning further enhanced the performance of some models, particularly the Decision Tree Regressor and Support Vector Regressor.

## Conclusion

This project demonstrates the application of machine learning for car price prediction. 
The Random Forest Regressor proved to be the most effective model for this task. 
The insights gained from this project will help us understand market dynamics and make informed decisions about product design, marketing, and pricing.

## Dependencies

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib

## Usage

If you prefer to run it in Google Colab, simply click the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aneeshmurali-n/Machine-Learning-for-U.S.-Car-Market-and-Pricing-Predictions/blob/main/U_S_Car_Market_and_Pricing_Predictions.ipynb) 

&nbsp;**OR**
1. Clone the repository.
2. Install the required dependencies.
3. Run the Jupyter Notebook.


## License

This project is licensed under the MIT License

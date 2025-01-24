# Comparative Performance Analysis of Machine Learning Regression Models on the Advertising Dataset
This repository explores the application of several machine learning regression models to the Advertising dataset, which examines the relationship between advertising spending (TV, Radio, Newspaper) and product sales.  The project aims to identify the model that offers the best predictive performance on unseen data while considering factors like interpretability and complexity.

* Linear Regression:
A simple yet interpretable model that achieved an R² score of 0.897, indicating a strong linear relationship between advertising budgets and sales.

* Decision Tree Regressor:
A non-linear model that captures complex patterns in the data, achieving an R² score of 0.920. While the model performed well, it may risk overfitting on larger datasets.

* Random Forest Regressor:
This ensemble-based method provided the best performance, with an R² score of 0.945, thanks to its ability to reduce variance and handle non-linearities effectively.

* Support Vector Regressor (SVR):
SVR achieved an R² score of 0.882, making it a decent option but slightly less effective than Random Forest or Decision Tree for this dataset.

* Ridge Regression:
By incorporating regularization to reduce overfitting, Ridge Regression achieved an R² score of 0.893, similar to Linear Regression but with better stability on unseen data.

This project also includes detailed evaluations, such as residual analysis, cross-validation results, and error metrics (MAE, MSE), to provide a comprehensive understanding of each model's strengths and limitations. The findings serve as a guide for selecting the most suitable regression model depending on specific business needs and data characteristics.

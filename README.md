# Prueba de Ironhack para competir en kaggle.
## Description:

This project is based on a kaggle competition (https://www.kaggle.com/c/diamonds-datamad0820) where students from the data analysis bootcamp compete to obtain the best score at trying to predict the price of a diamond base on its characteristics. This model is a supervised machine learning model where we obtain a dataset with the following features:

    id: only for test & sample submission files, id for prediction sample identification
    price: price in USD
    carat: weight of the diamond
    cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
    color: diamond colour, from J (worst) to D (best)
    clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
    x: length in mm
    y: width in mm
    z: depth in mm
    depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
    table: width of top of diamond relative to widest point (43--95)

In my project, 4 different models were created in order to obtain the best possible score, which are:

    Linear Regression model
    Random Forest model

In the file analisis.ipynb you can find the cleaning and model creation code that I developed this weekend, that goes from transforming strings to numerical data, to dropping columns and obtaining the clean dataset needed for the model training. The best score I could obtain was a root mean squared error of 528,23.

Documentation:

    https://scikit-learn.org
    https://pandas.pydata.org/
    https://medium.com/analytics-vidhya/solving-our-python-basketball-model-d89edfb83f79
    https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#objects-in-this-module

I really enjoyed making these models and competing to see who could obtain the best, although I would have loved to have more time in order to top myself on my hightest score. Hope you enjoy!

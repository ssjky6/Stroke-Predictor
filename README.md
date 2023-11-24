# Stroke Prediction using Machine Learning

[![Python Versions](https://img.shields.io/pypi/pyversions/yt2mp3.svg)](https://pypi.python.org/pypi/yt2mp3/)

**[Google Colaboratory](https://colab.research.google.com/drive/1rH2oucJ6namyM5m8nJJ4Tvy1W9uQ9kLy?usp=sharing)**

The data for training the model was obtained from [here](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset?select=healthcare-dataset-stroke-data.csv).
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Stroke is a blood clot   or bleeds in the brain, which can make permanent damage that has an effecton mobility, cognition, sight or communication. Stroke is considered as medical  urgent situation   and can cause long-term neurological damage, complications and often death. 

This web app is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

The web app was built in Python using the following libraries and deployed in Streamlit environment.
* streamlit
* pandas
* numpy
* scikit-learn
* pickle
* Imbalanced Learn

### We Experimented Outputs with Different Models: ###

* Random Forest Classifier

| Metrics  |  Score |
| :------------: | :------------: |
| accuracy_score  |  0.9336505079079337|  
|  precision_score | 0.9133935106123445  |
|  recall_score |  0.9587708066581306 |
|  f1_score |  0.9355322338830585 |

* Logistic Regression

| Metrics  |  Score |
| :------------: | :------------: |
| accuracy_score  |  0.8145814581458146| 
|  precision_score | 0.8058107772535387  |
|  recall_score |  0.8309859154929577 |
|  f1_score |  0.8182047402924861 |

* Adaboost Classifier

| Metrics  |  Score |
| :------------: | :------------: |
| accuracy_score  |  0.9324932493249325| 
|  precision_score | 0.9121951219512195  |
|  recall_score |  0.9577464788732394 |
|  f1_score |  0.9344159900062461 |

* SGD Classifier

| Metrics  |  Score |
| :------------: | :------------: |
| accuracy_score  |  0.7985084222707985| 
|  precision_score | 0.8023797206414899  |
|  recall_score |  0.7943661971830986 |
|  f1_score |  0.7983528503410113 |

* K Neighbours Classifier

| Metrics  |  Score | 
| :------------: | :------------: |
| accuracy_score  |  0.841969911276842|
|  precision_score | 0.7978628673196795  |
|  recall_score |  0.917797695262484 |
|  f1_score |  0.8536382041205192 |

* Voting Classifier

| Metrics  |  Score | 
| :------------: | :------------: |
| accuracy_score  |  0.8887745917448888|
|  precision_score | 0.855638745905479  |
|  recall_score |  0.9364916773367478 |
|  f1_score |  0.8942413497982639 |

* Neural Networks

| Metrics  |  Score | 
| :------------: | :------------: |
| accuracy_score  |  0.9261926192619262|
|  precision_score | 0.9045421423366529  |
|  recall_score |  0.9536491677336748 |
|  f1_score |  0.9284467713787086 |

### Literature References ###
* Artificial Neural Network Application to the Stroke Prediction. Available from: https://ieeexplore.ieee.org/document/9203638
* Performance Analysis of Machine Learning Approaches in Stroke Prediction. Available from: https://ieeexplore.ieee.org/document/9297525

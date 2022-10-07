# Credit Risk Analysis

## Overview (Supervised Machine Learning)

Machine Learning uses statistical algorithms to automate analytical model building, and uses these algorithms to learn from data, identify patterns, and provide predictions. The type of machine learning used for this project focuses on supervised machine learning model, in which labeled data is algorithmically analyzed, where patterns are identified, and then uses those previously learned patterns, to predict outcomes when that model is used on a new dataset. The predicted outcome is known as the 'Target'. 

A credit card company wants to use machine learning to predict credit risk, and believes that machine learning can help to identify a quicker and more reliable loan experience. The other positive to machine learning, is that it has the ability to provide more accurate results when it comes to identifying good candidates for loans, and try to prevent giving loans to the undesirable candidates, that could lead to higher loan default rates. In this project, I had to utilize different techniques to train and evaluate models with unbalanced credit risk classes, in which good loans outnumber risky loans. 

Using imbalanced-learn and scikit-learn libraries, I was able to build and evaluate multiple models using different resampling methods. 

1. Oversample the data using RandomOverSampler and SMOTE algorithms
2. Undersample the data using ClusterCentroids algorithm
3. Combinatorial approach of over- and undersampling using the SMOTEENN algorithm
4. Compare 2 machine learning models that reduce bias to predict credit risk: BalancedRandomForestClassifier and EasyEnsembleClassifier

## Resources 
* Python 3.9.12
* Anaconda (Jupyter Notebook & Pandas)
* Scikit-Learn (LogisticRegression, train_test_split, confusion_matrix, balanced_accuracy_score, classification_report_imbalanced)
* NumPy
* Pathlib 
* Imblearn (classification_report_imbalanced, BalancedRandomForestClassifier, EasyEnsembleClassifier, Cluster Centroids, RandomOverSampler, SMOTE, SMOTEENN)
* Dataset: [LoanStats_2019Q1.csv](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)

## Results 

### Table Results of All Machine Learning Algorithm Methods

| Model Used                     | Accuracy Score  | Precision | Recall | F1 |
| :------------------------------| --------------: | ---------:| ------:|---:|
| RandomOverSampler              | 0.63            | 0.99      | 0.69   |0.81|
| SMOTE                          | 0.62            | 0.99      | 0.64   |0.78|
| ClusterCentroids               | 0.51            | 0.99      | 0.45   |0.62|
| SMOTEENN                       | 0.64            | 0.99      | 0.58   |0.73|
| BalancedRandomForestClassifier | 0.78            | 0.99      | 0.91   |0.95|
| EasyEnsembleClassifier         | 0.92            | 0.99      | 0.94   |0.97|

* Accuracy Score: evaluates the model's performance or percentage of predictions that are correct. 
  * compares the actual outcome(y) values from the test set against the model's predicted values
* Precision: measure of how reliable a positive classification is. 
  * Precision = (TP) / (TP + FP)
  * TP = True Positives
  * FP = False Positives
* Recall(Sensitivity): metric used for evaluating a model's ability to predict the true positives of each available category.
  * Sensitivity = TP / (TP + FN)
  * FN = False Negatives
* F1-score: combines the precision and recall of a classifier into a single metric by taking their harmonic mean. 
  * F1 Score ranges from 0 - 1.
  * Value of 1 (High F1 Score): model perfectly classifies each observation into the correct class
  * Value of 0 (Low F1 Score): model is unable to classify any observation into the correct class

1. Oversample the data using RandomOverSampler and SMOTE algorithms

   * RandomOverSampler 

   ![RandomOverSampler](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/RandomOverSampler.png)

   * The accuracy score = 0.63, which means that the model was correct 63% of the time. 
   * The recall score for low-risk = 0.69, which means that the good candidate loans were correctly assessed 69% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 31% of the time. 
   * The recall score for high-risk = 0.57, which means that the bad candidate loans were correctly assessed 57% of the time. 
   * The low-risk F1 score = 0.81, because precision was high and recall(sensitivity) was relatively high, resulting in a higher F1 score. 
   * The high-risk F1 score = 0.02, because precision was extremely low and recall was not super low, but not super high either, resulting in a low F1 score. 

   * SMOTE 

   ![SMOTE](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/SMOTE_oversampling.png)

   * The accuracy score = 0.62, which means that the model was correct 62% of the time. 
   * The recall score for low-risk = 0.64, which means that the good candidate loans were correctly assessed 64% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 36% of the time. 
   * The recall score for high-risk = 0.61, which means that the bad candidate loans were correctly assessed 61% of the time.
   * The low-risk F1 score = 0.78, because precision was high and recall(sensitivity) was relatively high, resulting in a medium F1 score. 
   * The high-risk F1 score = 0.02, because precision was extremely low and recall was not super low, but not super high either, resulting in a low F1 score. 

2. Undersample the data using ClusterCentroids algorithm

   * ClusterCentroids 

   ![ClusterCentroids](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/ClusterCentroids.png)

   * The accuracy score = 0.51, which means that the model was correct 51% of the time. 
   * The recall score for low-risk = 0.45, which means that the good candidate loans were correctly assessed 45% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 55% of the time. 
   * The recall score for high-risk = 0.57, which means that the bad candidate loans were correctly assessed 57% of the time. 
   * The low-risk F1 score = 0.62, because precision was high and recall(sensitivity) was relatively low, resulting in a medium F1 score. 
   * The high-risk F1 score = 0.01, because precision was extremely low and recall was not super low, but not super high either, resulting in a low F1 score. 

3. Combinatorial approach of over- and undersampling using the SMOTEENN algorithm

   * SMOTEENN

   ![SMOTEENN](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_over_undersampling.png)

   * The accuracy score = 0.64, which means that the model was correct 64% of the time. 
   * The recall score for low-risk = 0.58, which means that the good candidate loans were correctly assessed 58% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 42% of the time. 
   * The recall score for high-risk = 0.70, which means that the bad candidate loans were correctly assessed 70% of the time. 
   * The low-risk F1 score = 0.73, because precision was high and recall(sensitivity) was higher than what would have been considered low, but it wasn't super high, resulting in an relatively higher F1 score. 
   * The high-risk F1 score = 0.02, because precision was extremely low and recall was relatively higher, resulting in a low F1 score. 


4. Compare 2 machine learning models that reduce bias to predict credit risk: BalancedRandomForestClassifier and EasyEnsembleClassifier

   * BalancedRandomForestClassifier 

   ![BRF](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/RandomForestClassifier.png)

   * The accuracy score = 0.78, which means that the model was correct 78% of the time. 
   * The recall score for low-risk = 0.91, which means that the good candidate loans were correctly assessed 91% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 9% of the time. 
   * The recall score for high-risk = 0.67, which means that the bad candidate loans were correctly assessed 67% of the time. 
   * The low-risk F1 score = 0.95, because precision was high and recall(sensitivity) was really high, resulting in an F1 score close to 1, indicating an almost perfect model in detecting low-risk loan candidates.
   * The high-risk F1 score = 0.07, because precision was extremely low and a medium recall score, resulting in a lower F1 score. 

   * EasyEnsembleClassifier 

   ![Ensemble](https://github.com/Lucky777b/Credit_Risk_Analysis/blob/main/Resources/EasyEnsembleAdaBoostClassifier.png)

   * The accuracy score = 0.92, which means that the model was correct 92% of the time. 
   * The recall score for low-risk = 0.94, which means that the good candidate loans were correctly assessed 94% of the time, and what should have been assessed as bad candidate loans were assessed as good candidate loans 6% of the time. 
   * The recall score for high-risk = 0.91, which means that the bad candidate loans were correctly assessed 91% of the time. 
   * The low-risk F1 score = 0.97, because precision was high and recall(sensitivity) was high as well, resulting in an F1 score close to 1, indicating an almost perfect model in detecting low-risk loan candidates. 
   * The high-risk F1 score = 0.14, because precision was low and recall score was extremely high, resulting in a lower F1 score. This F1 score was highest comparatively to the rest of the 5 models. 


   -- possible overfitting? Moreover, an extremely high metric should raise your suspicion of overfitting. Overfitting refers to an instance in which the patterns picked up by a model are too specific to a specific dataset

## Summary 

All of the models resulted in a average precision score of 0.99, or 99% precision, which means that all of the good candidate loans that were actually good candidate loans were identified 99% of the time, which is good when looking at the true positives. One downfall to having such a high precision score, is that it can result in higher false positives, or the bad candidate loans who should not have been predicted as good candidate loans, but were. As shown in the output images above, the result of precision for the high-risk loans in all of the models was pretty low. This could be problematic for a credit card company, who is trying to decrease the amount of the defaulted loans, and that is why it was important to perform multiple machine learning models, instead of just one. It is understandable that one might never be able to correctly predict the good versus bad candidate loans 100% of the time, but it is important that the good candidate loans are predicted correctly almost 100% of the time, and it is also just as important for the high risk, or bad candidate loans, to be identified at a high rate as well. 

If the model was designed to have a higher sensitivity rate over a higher precision rate, this could have lead to those low-risk candidates being incorrectly predicted to be high-risk candidates at a higher rate, and resulting in a higher amount of false negatives, but decreasing the amount the defaulted loans, or false positives. As previously discussed, a high precision rate can also result in a higher number of false positives, which is another tradeoff of high precision over high sensitivity. 

The precision and F1 scores, for high-risk loans, were low in all models, indicating an imbalance between precision and sensitivity scores. Even though the scores were low in all of the models, it is still important to determine which model works the best for determining future good or bad loan candidates. The model with the highest precision rate was the EasyEnsembleClassifier, indicating that this model would predict high-risk and low-risk candidates more reliably than the other 5 models.


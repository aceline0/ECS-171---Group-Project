# Credit Card Fraud Detection

ML project classifying credit card transactions as real or fake using [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Ensure that you have GIT LFS installed in order to download the dataset.

# Project structure

The file `data/creditcard.csv` contains the dataset with all 285.000 transactions. 

The folder `code/` contains the different notesbooks and python scripts for the project.

Inside `assets/img/` assets in the form of images related to the different models can be found.

# Introduction

Sadly, finance and fraud are two words that are deeply intertwined. When there is substantial monetary gain available, malicious actors will attempt to carve out a piece. Ever since the Diner Club's card in 1951, the convenience presented by credit cards has led to substantial growth, so much so that in 2021, 581 billion credit card transactions were processed.  

When choosing a dataset to work on, it seemed natural to pick an area with which we were familiar. Since almost everyone carries a credit card, both interpreting the data and understanding important metrics to measure came naturally. What is most fascinating with the dataset is its real-world applicability, meaning what we are learning in this course is being applied to solve a real (big!) problem. However, it comes with its trade-offs, mainly that all features are anonymized, named v1-v28, making it more difficult to draw conclusions.  

The impact of implementing and having good models to detect financial fraud is broad. If a model can detect a criminal's transactions, huge societal problems, e.g., money laundering, can be thwarted. Furthermore, fraudulent transactions are a cost for the banks/institutions handling it, meaning it shows up as a cost on their income statement, something more often than not pushed down to consumers (us), in the form of higher transaction fees / higher interest rates etc.  

Finally, with the world becoming increasingly digitalised, more data can be tied to each transaction. In machine learning, more data points and more features allows for better models, i.e., better detection of fraud. The future of finance will no doubt continue to become more data driven, which is why this project serves as a solid stepping stone for understanding how machine learning ties into tomorrow's systems.  

# Methods

### Data Exploration
* **Data Overview**\
Firstly, the data was explored, identifying the number of features, what they mean, what our dependent variable should be, and if there's any imbalance.
* **Duplicate Data Entries**\
Secondly, the dataset was checked for duplicate entries, and removed if any were found.
* **Null Data**\
Thirdly, the dataset is checked for null values, and if encountered, the affected rows are dropped.

### Data Preprocessing
* **Data Scaling**\
Firstly, every feature was investigated to see if any transformation was needed. Afterwards, using plots of each feature, it's decided what type of scaling to use, either standardization or min-max scaling.
* **Feature Selection**\
Secondly, kde-plots were used to determine how well each feature distinguished fraud and non-fraud. Furthermore, a correlation matrix was used to find the correlation between variables, and thereupon deciding to drop features not contributing to the prediction.
* **Train Set, Test Set**\
Lastly, the dependent value's frequency is investigated, to decipher if over/undersampling would be needed before training a model.
The preprocessing was added to a seperate file as running the data exploration in the beginning of working on a new model inefficient.

### Models

* **Model 1 - Logistic Regression**\
The first model was [Logistic Regression](./code/firstModel.ipynb), with the hyperparameter max_iterations set to 1200. Two methods of altering complexity were used to find differences in the model's performance:
  1. PCA - decreasing complexity through dimensionality reduction.
  2. Polynomial features - adding complexity by introducing more features.

* **Model 2 - K-Nearest Neighbors**\
The second model used was [K-Nearest Neighbors](code/knn.ipynb) (KNN). Two parameters were investigated to plot fitting graphs:
  1. PCA - decreasing complexity through dimensionality reduction.
  2. K-value - testing different k-values (different number of neighbors)

* **Model 3 - Gaussian Naive Bayes**\
The third model used was [Gaussian Naive Bayes](code/NaiveBayes.ipynb). No complexity analysis was done for this model.

### Evaluation
In evaluating the train and test set, two methods of measuring model accuracy were used:
  1. Confusion Matrix
  2. Classification Report

This was done using the built in methods in sklearn.metrics, as demonstrated below:
```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, yhat))
cm = confusion_matrix(y_test, yhat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-fraud', 'Fraud'])
disp.plot()
```

# Results

### Data exploration
* **Data Overview**

  ![Fraud count in the dataset](/assets/img/data_exp/fraud_count.png)

  The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. \
  It contains 30 numerical input variables (V1, V2 ??? V28, Time, Amount). \
  The variable ???Class??? is the dependent variable (1 = fraud, 0 = not fraud) our model will predict.

* **Duplicate Data Entries**\
  There were 1081 duplicate samples, all of which were removed.

* **Null Data**\
  No null data was encountered, therefore, no rows were dropped in this stage.

### Preprocessing
* **Data Scaling**

  It was found that the features ???Time??? and ???Amount??? have large numerical values, which are different from other features. \
  ???Time??? is the number of seconds elapsed between this transaction and the first transaction in the dataframe, spanning 48 hours.  Therefore, another column ???Hour??? was added, based on ???Time???, as ???Hour??? shows the peak period of credit card usage and its relation to the time of credit card fraud.
  Since the data is not normally distributed, min-max normalization on the data was done, resulting in the variables now having values between 0 and 1, simplifying further analysis.

* **Feature Selection**

  Because of the imbalanced nature of the data, both an undersampled and oversampled correlation matrix were investigated. It was found that they were similar, i.e., sampling had no affect on correlation between variables.
  
  ![Correlation Matrix](assets/img/data_exp/feature_corr.png)
   
   Using kde plots and correlation matrices, it was inferred that Time, V13, V15, V22, V23,V24,V25,V26, V27,V28, Amount, V8, V21 cannot distinguish fraud cases and non-fraud cases well, and thus were dropped.

* **Train Set, Test Set**

  As previously mentioned, there's way more non-fraud data than fraud data. We choose to oversample the fraud data here. If we undersample, we will suffer the risk of losing important information since undersample means we only utilize a tiny fraction of the non-fraud data. In addition, compared to the whole dataset, the minority class does not have sufficient size. Therefore, we will oversample.

### Model 1 - Logistic Regression
* **Comparing Train and Test Error**

  ![Logistic Regression Training Confusion Matrix](/assets/img/log_reg/log_reg_training_matrix.png)

  In the classification report for the training data, the precision was lower (91% compared to 97%) for class 0, whilst recall was higher (97% vs 90% for class 1). Regarding the f1-score they remained largely the same, with scores of 94% and 93% respectively.

  ![Logistic Regression Training Confusion Matrix](/assets/img/log_reg/log_reg_test_matrix.png)

  Meanwhile, the test data's classification report showed a large difference in precision and recall for the two classes. Class 0 had a precision of 100% and a recall of 97%, whilst class 1 had a precision of 4% with a recall of 86%. The f1-scores were 98% and 8% respectively.

* **The model's position on a fitting graph**

  As mentioned in the Methods section, two methods of altering complexity were used two find differnences in the model's performance: PCA, and Polynomial features.

  ![Logistic Regression PCA fitting graph](/assets/img/log_reg/logistic_regression_pca.png)

  From the PCA-analysis, the fitting graph shows that using 7 principal components decreased the model's complexity adequatly whilst still retaining a low training/test loss.

  ![Polynomial Features fitting graph](/assets/img/log_reg/logistic_regression_poly_fitting.png)

  Using different degrees of polynomial features, it became evident that adding complexity using polynomial features only decreased training loss with test loss remaining largely the same. 

### Model 2 - K-nearest neighbors
* **Comparing Train and Test Error**

  Starting with the training data, the model achieved a precision, recall and f1-score of 1.0 for class 0. For class 1, on the other hand, the values were 0.92, 0.78 and 0.85 respectively.\
  Moving on with the test data, precision, recall and f1-score for class 0 remained the same. However, precision went up to 0.99, recall down to 0.76 and in total the f1-score moved up to 0.86. The confusion matrix below shows the results of predicting the test data:

  ![KNN test confusion matrix](/assets/img/knn/knn_test_confusion_matrix.png)

* **The model's position on a fitting graph**

  Starting with PCA, it was found that 13 was the optimal number of principal components (as seen in the fitting graph below).

  ![PCA fitting graph](/assets/img/knn/pca_fitting.png)

  Regarding the optimal k-value from a fitting-perspective, the value 7 optimized the model's performance.

  ![K-value optimization fitting graph](/assets/img/knn/kvalue_fitting.png)

### Model 3 - Naive Bayes
* **Comparing Train and Test Error**

  In the classification report for both training data and test data, Class 0 and Class 1 had varying scores for precision, recall, and f1-score. For the train data these were 98%, 84% and 90% respectively. For the test data we had 6% precision, 77% recall and 11% f1-score. The accuracy score for both the training data and test data is similar with the accuracy of the training data being 91% and 98% for test data. 

  ![Confusion matrix NB test data](/assets/img/nb/cm_test.png)


# Discussion
### Data exploration
During the exploration stage, it was found that we had no null data. This is of course possible, however, one could say that it is unlikely. Real-world data is messy, and this dataset had already been processed (with the null data dropped). However, encountering duplicate data entries, and thereby having to remove these, strengthened the validity of the dataset. This was done since multiple identical samples can lead to a biased model, something you always want to lower in a model. Finally, the anonymization of the feature names increased the difficulty of drawing conclusions from our findings. However, this is expected since it is real world data, and consumers privacy is in question. 

### Preprocessing
In this step, by transforming the 'Time' category, we were able to find out (and model upon) if frauds were more likely to happen at a specific time of the day. In contrast to the data exploration, this showed the thesis that "real data is messy", but that clever transformations can make it "less messy" and actually useful!

Regarding the oversampling, 'SMOTE' was used, mainly beacuse it is a type of sampling that may achieve a higher recall. Recall is an important performance metric with our model since we want to detect as many fraud cases as possible to protect people's money. It is awful if our model identifies a fraud case as a non-fraud case, because this will lead to people losing money and having to contact their payment processor to get it resolved. Finally, this oversampling was only conducted post train test split, since testing the model should be done on unseen data, i.e., non-oversampled.

### Logistic regression
Firstly, introducing polynomial features was deemed inadequate, as it increases the model's risk of overfitting (pushing the model further out on the x-axis of a fitting graph).

The stark difference in error can be explained by the imbalanced nature of the input data, i.e., it contains way more non-fraudulent transactions than fraudulent ones.

With a high recall being the most important metric in evaluating the model, and the test data having a recall of 84%, the conclusion was drawn that the model fairly good at classifying fraudulent transactions, however, not optimal for the classification problem at hand.

### K-nearest neighbors
As the results presented, the model had a 99% precision on test data, with only one false positive (meaning this figure was obtained by taking 1/(68+1)). This shows the problem presented by unbalanced data, especially with very few positives of our dependent variable. As mentioned in the preprocessing discussion, this ties back to real data being messy, and the model did the best it could with the data at hand.

With a test recall of 0.76 compared to 0.86 for logistic regression, shows it is not as suited for this type of classification. However, the stark increase in precision is impressive, and could probably same time in overhead (meaning, not as many false positives, implies less work for the department investigating all flagged transactions).

### Naive Bayes
Since the model has a 77% test recall on class 1, it was able to identify a significant amount of the fraudulent transactions. However with a precision of .06 it is simply finding a lot of false positives, since only 6% of classified frauds are actually true positives. The model would likely be more effective if given more fraudulent data to train on, however as it currently stands, it would not be effective in production since it would produce too many false positives.

Comparing this naive bayesian to our previous models, this model had the lowest recall score. The model is therefore less suited for predicting credit card fraud. On the other hand, the precision is higher than the logistic model (4%) but a lot less than for KNN (26%). 

# Conclusion
First of all, since none of the models provided a high recall (the highest being 0.86 with the logistic regression), we believe there might is model out there that is better suited for the problem at hand - and that we simply have not found it. --- However, our logistic model can be used in the awarness of the model evaluation. High recall is crucial when, as mentioned, the consequences for predicting non-fraud for a transaction that is actually fraud are has a higher cost than predicting fraud when there is no fraud. ----From this, there are quite a few future direction that could improve the results of this study.

Firstly, trying out more models, e.g., decision tree, neural networks, etc., would greatly increase the odds of finding one that could achieve a higher recall. With neural networks, multiple hyperparameters can be tuned (number of hidden layers, activation functions, number of nodes, batch size, number of epochs), where one combination of these would probably yield a better result than we found.

Secondly, time could be spent improving the dataset at hand. As previously discussed, the unbalanced nature of the dataset impacted the models' performances. If possible, finding a dataset with many more fraudulent transactions would give a better foundation upon which research could be conducted.

Thirdly, an ensamble of models could be tried, to see if we can get the precision of the KNN, and the recall of the logistic regression (or some other models with better accuracy) From the concatenation of these predictions, we could obtain a new predicted value resulting in a better performance.

To round up, this project opened our eyes to problems data scientists encounter in the real world, exposing us to nuances in machine learning only listening to lectures and taking tests never would be able to depict properly. With these newfound insights, if we were to conduct research on another dataset, they would surely come in handy, saving time and allowing us to reach more useful conclusions in a shorter timeframe.

# Collaboration
First of all, a general collaboration statement: All individuals were active in the group discord, providing input when questions arose, discussing choices of models, meeting times, workload distribution, etc. No roles were assigned in the beginning of the project, however, each collaboration statement should serve as a pointer to each person's main focus.

* **Kevin Rasmusson (Writing, code contribution to KNN)**\
Writeup: Wrote introduction, discussion (not the part on naive bayes), conclusion, reformulations and contributions to the methods section, first model spell check. Added images, separated method/result/discussion for every subsection (preprocessing, exploration, models 1-3). Did the first draft of the KNN model (only training) and added comments to the code.
* **Celine Nygaard Weiseth (Writing, code contribution to Naive Bayes)**\
Wrote method and results for the naive bayes model, contribution on the naive bayes code part, added and fixed spelling in preprocessing, filled in on conclusion. Took initiative to create a discord server that worked well for communicating as all group member have been responsive. 
* **Andrea Rodriguez (Writing, Gave Feedback and Suggestions)**\
Organized and shared google docs for the writeup. Wrote train and error results for some models. Gave suggestions to use naive bayes on the last model and also gave feedback when needed. 
* **Hongye Xu (Writing, code contribution to data exploration, processing and fitting graph of the model)**\
Coded and wrote the documentation of the data exploration and processing step, contributed ideas on and coded how to increase/decrease the complexity of the model, and coded the fitting graph.
* **Javier Sande (Code contribution to processing, Logistic Regression and KNN)**\
Coded the training of the first model and the optimization and visualization methods in the first model and KNN. Gave and implemented suggestions on how to optimize the results by tuning parameters. Gave suggestions on the interpretation of the results and how they might be applied to the next steps.
* **Aske Nord Raahauge (Writing, code contribution to Naive Bayes)**\
Wrote initial preprocessing of data & performed bugfixing on Naive Bayes in order to make it work, also cleaned it up and added assets.
Made seperate preprocessor for easy running of all models. Got github initially up and running with code, data and writeup.

# References
* https://stackoverflow.com/questions/55104819/display-count-on-top-of-seaborn-barplot
* https://seaborn.pydata.org/generated/seaborn.kdeplot.html
* https://stackoverflow.com/questions/69513501/seaborn-plot-displot-with-hue-and-dual-y-scale-twinx
* https://stackoverflow.com/questions/24500065/closing-matplotlib-figures
* https://towardsdatascience.com/smote-fdce2f605729

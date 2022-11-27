# Credit Card Fraud Detection

ML project classifying credit card transactions as real or fake using [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Ensure that you have GIT LFS installed in order to download the dataset.

# Project structure

The file `data/creditcard.csv` contains the dataset with all 285.000 transactions. 

The folder `code/` contains the different notesbooks and python scripts for the project.

# Data Exploration and Processing
* **Data Overview**\
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. \
It contains 30 numerical input variables (V1, V2 … V28, Time, Amount). \
The variable ‘Class’ is the dependent variable (1 = fraud, 0 = not fraud) our model will predict.
* **Duplicate Data Entries**\
There are 1081 duplicate samples. We will remove them because multiple identical samples can lead to biased model, which favors this subset of samples.
* **Null Data**\
Check null values because they don’t contribute to build the model and they can affect the performance. Fortunately, there is no null data.
* **Data Scaling**\
The data ‘Time’ and ‘Amount’ have large numerical values, which are different from other features. \
‘Time’ is the number of seconds elapsed between this transaction and the first transaction in the dataframe, spanning into 48 hours. We decide to add another column ‘Hour’ based on ‘Time’, as ‘Hour’ may show the peak period of credit card usage and its relation to the time of credit card fraud.\
Since the data are not normally distributed, we will perform normalization on the data.
* **Feature Selection**\
Since we have 31 features now.  We want to see how well these features help the model distinguish fraud and non-fraud.\
We firstly use kde plots to visualize the each feature's distribution to fraud cases or non-fraud cases.\
Secondly, we use correlation matrix plot to find the correlation between variables. Notice that  we have imbalance data about ‘Class’ in the original dataframe, the correlation matrix may come out biased or inaccurate. So we perform undersample and oversample on the dataframe, and then do the correlation matrix plot on both sampled dataframe. We want to how undersample and oversample will affect the correlation matrix.\
Using kde plots and correlation matrix plots, we infer that Time, V13, V15, V22, V23,V24,V25,V26, V27,V28, Amount, V8, V21can not distinguish fraud cases and non-fraud cases well, and thus we will drop these features.
* **Train Set, Test Set**\
Remember that we have much more fraud data than non-fraud data. We choose to oversample the non-fraud data here. If we undersmaple, we will suffer the risk of losing important information since undersample means we only utilize a litte bit of the non-fraud data. In addition, compared to the whole dataset, the minority class does not have sufficient size. Therefore, we will oversample. Specifically, we will use 'SMOTE' (from online resources, ‘SMOTE’ may achieve higher recall). Recall is a good performance metric to our model because we want to detect as many fraud cases as possible to protect people's properties. It is awful if our model identifies a fraud case as a non-fraud case, and then people will lose money and they may need to contact the bank for further actions. \
Notice that we will oversample the train set after train test split because we want to test the model on UNSEEN test data.

# Model Building and Evaluation
## First Model
For our first model, we chose Logistic Regression. We started with 1000 max_iterations. 

In evaluating the train and test set, we chose two methods of measuring model accuracy by using confusion matrix and classification report. 

* **Train Results**\

* **Test Results**\


* **Comparing Train and Test Error**\








Reference:
* https://stackoverflow.com/questions/55104819/display-count-on-top-of-seaborn-barplot
* https://seaborn.pydata.org/generated/seaborn.kdeplot.html
* https://stackoverflow.com/questions/69513501/seaborn-plot-displot-with-hue-and-dual-y-scale-twinx
* https://stackoverflow.com/questions/24500065/closing-matplotlib-figures
* https://towardsdatascience.com/smote-fdce2f605729

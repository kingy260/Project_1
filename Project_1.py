import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV

"STEP 1:"

#Set up data frame from csv
df = pd.read_csv("Project_1_Data.csv")
print(df.info())

#Split data, 80/20 train to test ratio, using stratified sampling
#Done before data visualization to avoid snooping bias

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 69)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
    
# Variable Selection, drop step column from x (features) and make y (outcome) the step
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

"STEP 2:"
#visualization of only training data to avoid snooping bias

#scatter matrix to see relationships between each variable and histograms
pd.plotting.scatter_matrix(strat_df_train)


"STEP 3:"
#Correlation Analysis

# Create the correlation matrix heatmap
# Although outcome is categorical, since it is represented by discrete numbers
# we can look at the heatmap to see any direct correlations bw coordinates and step.
# Also assisted by the scatter matrix
plt.figure()
corr_matrix = (strat_df_train).corr()
sns.heatmap(np.abs(corr_matrix))
# Most importantly, can see no significant correlation between features.


"STEP 4:"
# Scaling
#set up scaler object
my_scaler = StandardScaler()
#calculating mean and std deviation
my_scaler.fit(X_train.iloc[:,0:-1])

#scale the train data, dont scale the "step" column (categorical)
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:-1])

#since the scaler outputs a numpy array, recreate the data frame and join with the step column
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns[0:-1])
X_train = scaled_data_train_df.join(X_train.iloc[:,-1:])

#Use mean and std deviation from training data to scale testing data as well, avoiding a data leak
scaled_data_test = my_scaler.transform(X_test.iloc[:,0:-1])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns[0:-1])
X_test = scaled_data_test_df.join(X_test.iloc[:,-1:])

#LOGISTIC REGRESSION

#parameter grid that will be used in gridsearch for LR. 
#The main params to tune are the solver, penalty, and C (inverse of regularization strength, smaller C = stronger reduction in overfitting)
LRparam_grid = {
     'solver': ['lbfgs', 'newton-cg', 'sag'],
     'penalty': [None, 'l2'],
     'C': [0.01,0.1, 1, 10]
 }

#initialize logistic regression
LRmodel = LogisticRegression(multi_class='multinomial', random_state=69, max_iter=100)
LRgrid_search = GridSearchCV(LRmodel, LRparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
LRgrid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", LRgrid_search.best_params_)

#get the best model
LRmodel = LRgrid_search.best_estimator_

#use the best estimator on the testing data
LRy_pred = LRmodel.predict(X_test)
LRy_pred_train = LRmodel.predict(X_train) 

#Metrics
print("\nClassification Report:")
print(classification_report(y_test, LRy_pred))
print("Accuracy Score:", accuracy_score(y_test, LRy_pred))
print("Training Accuracy Score:", accuracy_score(y_train, LRy_pred_train))

#make the confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, LRy_pred)

#create dataframe from confusion matrix so it can be turned into a heatmap
cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# #RANDOM FOREST

# #parameter grid that will be used in gridsearch for RF. 
# #The main params to tune are num_estimators max depth, min_samples split,... 
# #min_samples_leaf, criterion, and max features
# RFparam_grid = {
#      'n_estimators': [5, 10, 30, 50],
#      'max_depth': [None, 5, 15, 45],
#      'min_samples_split': [2, 5, 10],
#      'min_samples_leaf': [1, 2, 4, 6],
#      'max_features': [None,0.1,'sqrt', 'log2', 1, 2, 3],
#      'criterion': ['gini', 'entropy']
#  }

# #initialize random forest
# RFmodel = RandomForestClassifier(random_state=69)

# RFgrid_search = GridSearchCV(RFmodel, RFparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

# #do the grid search
# RFgrid_search.fit(X_train, y_train)

# #best parameters
# print("Best Parameters:", RFgrid_search.best_params_)

# #get the best model
# RFmodel = RFgrid_search.best_estimator_

# #use the best estimator on the testing data
# RFy_pred = RFmodel.predict(X_test)
# RFy_pred_train = RFmodel.predict(X_train) 

# #Metrics
# print("\nClassification Report:")
# print(classification_report(y_test, RFy_pred))
# print("Accuracy Score:", accuracy_score(y_test, RFy_pred))
# print("Training Accuracy Score:", accuracy_score(y_train, RFy_pred_train))

# #make the confusion matrix
# print("Confusion Matrix:")
# cm = confusion_matrix(y_test, RFy_pred)
# #create dataframe from confusion matrix so it can be turned into a heatmap
# cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
# #make heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False,
#             xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Random Forest Confusion Matrix')
# plt.show()

#SUPPORT VECTOR MACHINE

#parameter grid that will be used in gridsearch for RF. 
#The main params to tune are kernel, C, and gamma
SVMparam_grid = {
     'gamma': ['scale','auto',10,100],
     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'C': [0.001,0.01,0.1, 1, 10]
 }

#initialize SVM
SVMmodel = svm.SVC(random_state=69)

SVMgrid_search = GridSearchCV(SVMmodel, SVMparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
SVMgrid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", SVMgrid_search.best_params_)

#get the best model
SVMmodel = SVMgrid_search.best_estimator_

#use the best estimator on the testing data
SVMy_pred = SVMmodel.predict(X_test)
SVMy_pred_train = SVMmodel.predict(X_train) 

#Metrics
print("\nClassification Report:")
print(classification_report(y_test, SVMy_pred))
print("Accuracy Score:", accuracy_score(y_test, SVMy_pred))
print("Training Accuracy Score:", accuracy_score(y_train, SVMy_pred_train))

#make the confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, SVMy_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SVM Confusion Matrix')
plt.show()

#RANDOM FOREST WITH RANDOM SEARCH CV

#parameter grid that will be used in random search for RF. 
#The main params to tune are num_estimators max depth, min_samples split,... 
#min_samples_leaf, criterion, and max features
RF2param_grid = {
      'n_estimators': [5, 10, 15, 20, 25, 30, 50],
      'max_depth': [None, 5,10, 15, 25, 45],
      'min_samples_split': [2, 5, 10, 15],
      'min_samples_leaf': [1, 2, 4, 6, 12],
      'max_features': [None,0.1,'sqrt', 'log2', 1, 2, 3],
      'criterion': ['gini', 'entropy']
  }

#initialize random forest
RF2model = RandomForestClassifier(random_state=69)

RF2rand_search = RandomizedSearchCV(RF2model, RF2param_grid, n_iter = 10, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
RF2rand_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", RF2rand_search.best_params_)

#get the best model
RF2model = RF2rand_search.best_estimator_

#use the best estimator on the testing data
RF2y_pred = RF2model.predict(X_test)
RF2y_pred_train = RF2model.predict(X_train) 

#Metrics
print("\nClassification Report:")
print(classification_report(y_test, RF2y_pred))
print("Accuracy Score:", accuracy_score(y_test, RF2y_pred))
print("Training Accuracy Score:", accuracy_score(y_train, RF2y_pred_train))

#make the confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, RF2y_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest RandomSearch CV Confusion Matrix')
plt.show()
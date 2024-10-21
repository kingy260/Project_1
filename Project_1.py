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
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
import joblib

"STEP 1:"

#Set up data frame from csv
df = pd.read_csv("Project_1_Data.csv")
#print(df.info())

#Split data, 80/20 train to test ratio, using stratified sampling
#Done before data visualization to avoid snooping bias

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 74)
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
LRmodel = LogisticRegression(multi_class='multinomial', random_state=74, max_iter=5000)
LRgrid_search = GridSearchCV(LRmodel, LRparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
LRgrid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", LRgrid_search.best_params_)

#get the best model
LRmodel = LRgrid_search.best_estimator_

#use the best estimator on the testing and training data
LRy_pred = LRmodel.predict(X_test)
LRy_pred_train = LRmodel.predict(X_train) 

#Metrics
print("\nClassification Report:")

LR_report = classification_report(y_test, LRy_pred, output_dict=True)
LR_accuracy = accuracy_score(y_test, LRy_pred)
print("Accuracy Score:", LR_accuracy)
LR_trainingacc = accuracy_score(y_train, LRy_pred_train)
print("Training Accuracy Score:", LR_trainingacc)

#make the confusion matrix
print("Confusion Matrix:")
LRcm = confusion_matrix(y_test, LRy_pred)

#create dataframe from confusion matrix so it can be turned into a heatmap
LRcm_df = pd.DataFrame(LRcm, index=np.unique(y_test), columns=np.unique(y_test))

#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(LRcm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

LRaccuracy_df = pd.DataFrame({'Metric': ['Accuracy','Training Accuracy'], 'Score': [LR_accuracy, LR_trainingacc]})
LR_report_df = pd.DataFrame(LR_report).transpose()

#SAVE TO CSV
with open('LR_results.csv', 'w', newline='') as f:
    f.write("Classification report\n")
    LR_report_df.to_csv(f)
    f.write("\nAccuracy Score\n")
    LRaccuracy_df.to_csv(f, index=False)
    

#RANDOM FOREST

#parameter grid that will be used in gridsearch for RF. 
#The main params to tune are num_estimators max depth, min_samples split,... 
#min_samples_leaf, criterion, and max features
RFparam_grid = {
      'n_estimators': [5, 10, 30, 50],
      'max_depth': [None, 5, 15, 45],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4, 6],
      'max_features': [None,0.1,'sqrt', 'log2', 1, 2, 3],
      'criterion': ['gini', 'entropy']
  }

#initialize random forest
RFmodel = RandomForestClassifier(random_state=74)

RFgrid_search = GridSearchCV(RFmodel, RFparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
RFgrid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", RFgrid_search.best_params_)

#get the best model
RFmodel = RFgrid_search.best_estimator_

#use the best estimator on the testing and training data
RFy_pred = RFmodel.predict(X_test)
RFy_pred_train = RFmodel.predict(X_train) 

#Metrics

RF_report = classification_report(y_test, RFy_pred, output_dict=True)
RF_accuracy = accuracy_score(y_test, RFy_pred)
print("Accuracy Score:", RF_accuracy)
RF_trainingacc = accuracy_score(y_train, RFy_pred_train)
print("Training Accuracy Score:", RF_trainingacc)

#make the confusion matrix
print("Confusion Matrix:")
RFcm = confusion_matrix(y_test, RFy_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
RFcm_df = pd.DataFrame(RFcm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(RFcm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix')
plt.show()

RFaccuracy_df = pd.DataFrame({'Metric': ['Accuracy','Training Accuracy'], 'Score': [RF_accuracy, RF_trainingacc]})
RF_report_df = pd.DataFrame(RF_report).transpose()

#SAVE TO CSV
with open('rf_results.csv', 'w', newline='') as f:
    f.write("Classification report\n")
    RF_report_df.to_csv(f)
    f.write("\nAccuracy Score\n")
    RFaccuracy_df.to_csv(f, index=False)
    

#SUPPORT VECTOR MACHINE

#parameter grid that will be used in gridsearch for RF. 
#The main params to tune are kernel, C, and gamma
SVMparam_grid = {
     'gamma': ['scale','auto',10,100],
     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'C': [0.001,0.01,0.1, 1, 10]
 }

#initialize SVM
SVMmodel = svm.SVC(random_state=74)

SVMgrid_search = GridSearchCV(SVMmodel, SVMparam_grid, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
SVMgrid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", SVMgrid_search.best_params_)

#get the best model
SVMmodel = SVMgrid_search.best_estimator_

#use the best estimator on the testing and training data
SVMy_pred = SVMmodel.predict(X_test)
SVMy_pred_train = SVMmodel.predict(X_train) 

#Metrics
SVM_report = classification_report(y_test, SVMy_pred, output_dict=True)
SVM_accuracy = accuracy_score(y_test, SVMy_pred)
print("Accuracy Score:", SVM_accuracy)
SVM_trainingacc = accuracy_score(y_train, SVMy_pred_train)
print("Training Accuracy Score:", SVM_trainingacc)

#make the confusion matrix
print("Confusion Matrix:")
SVMcm = confusion_matrix(y_test, SVMy_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
SVMcm_df = pd.DataFrame(SVMcm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(SVMcm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SVM Confusion Matrix')
plt.show()

SVMaccuracy_df = pd.DataFrame({'Metric': ['Accuracy','Training Accuracy'], 'Score': [SVM_accuracy, SVM_trainingacc]})
SVM_report_df = pd.DataFrame(SVM_report).transpose()

#SAVE TO CSV
with open('SVM_results.csv', 'w', newline='') as f:
    f.write("Classification report\n")
    SVM_report_df.to_csv(f)
    f.write("\nAccuracy Score\n")
    SVMaccuracy_df.to_csv(f, index=False)


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
RF2model = RandomForestClassifier(random_state=74)

RF2rand_search = RandomizedSearchCV(RF2model, RF2param_grid, n_iter = 10, cv=5, scoring='f1_weighted', n_jobs=1)

#do the grid search
RF2rand_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", RF2rand_search.best_params_)

#get the best model
RF2model = RF2rand_search.best_estimator_

#use the best estimator on the testing and training data
RF2y_pred = RF2model.predict(X_test)
RF2y_pred_train = RF2model.predict(X_train) 

#Metrics
RF2_report = classification_report(y_test, RF2y_pred, output_dict=True)
RF2_accuracy = accuracy_score(y_test, RF2y_pred)
print("Accuracy Score:", RF2_accuracy)
RF2_trainingacc = accuracy_score(y_train, RF2y_pred_train)
print("Training Accuracy Score:", RF2_trainingacc)


#make the confusion matrix
print("Confusion Matrix:")
RF2cm = confusion_matrix(y_test, RF2y_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
RF2cm_df = pd.DataFrame(RF2cm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(RF2cm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest RandomizedSearchCV Confusion Matrix')
plt.show()

RF2accuracy_df = pd.DataFrame({'Metric': ['Accuracy','Training Accuracy'], 'Score': [RF2_accuracy, RF2_trainingacc]})
RF2_report_df = pd.DataFrame(RF2_report).transpose()

#SAVE TO CSV
with open('RF2_results.csv', 'w', newline='') as f:
    f.write("Classification report\n")
    RF2_report_df.to_csv(f)
    f.write("\nAccuracy Score\n")
    RF2accuracy_df.to_csv(f, index=False)


"STEP 6"

# stacking classifier

# The RandomizedSearchCV rf model and SVM model are stacked. This means that...
# the outputs of both are used as inputs to a final estimator, in this case
# a logistic regression model (default for stacking classifier)
final_model = LogisticRegression(max_iter=5000, solver='lbfgs')
stacked_rf2_svm =  StackingClassifier(estimators= [('rf',RF2model),('svm', SVMmodel)], cv=5, n_jobs=1, final_estimator = final_model)

#fit the stacked classifier
stacked_rf2_svm.fit(X_train, y_train)

#use the best estimator on the testing and training data
stackedy_pred = stacked_rf2_svm.predict(X_test)
stackedy_pred_train = stacked_rf2_svm.predict(X_train) 

#Metrics
stacked_report = classification_report(y_test, stackedy_pred, output_dict=True)
stacked_accuracy = accuracy_score(y_test, stackedy_pred)
print("Accuracy Score:", stacked_accuracy)
stacked_trainingacc = accuracy_score(y_train, stackedy_pred_train)
print("Training Accuracy Score:", stacked_trainingacc)


#make the confusion matrix
print("Confusion Matrix:")
stackedcm = confusion_matrix(y_test, stackedy_pred)
#create dataframe from confusion matrix so it can be turned into a heatmap
stackedcm_df = pd.DataFrame(stackedcm, index=np.unique(y_test), columns=np.unique(y_test))
#make heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(stackedcm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Stacked RF RandomizedSearchCV and SVM Confusion Matrix')
plt.show()

stackedaccuracy_df = pd.DataFrame({'Metric': ['Accuracy','Training Accuracy'], 'Score': [stacked_accuracy, stacked_trainingacc]})
stacked_report_df = pd.DataFrame(stacked_report).transpose()

#SAVE TO CSV
with open('stacked_results.csv', 'w', newline='') as f:
    f.write("Classification report\n")
    stacked_report_df.to_csv(f)
    f.write("\nAccuracy Score\n")
    stackedaccuracy_df.to_csv(f, index=False)


"STEP 7"
#save the stacked model as a pickle file
joblib.dump(stacked_rf2_svm, 'stacked_rf2_svm.pkl')

#load model from pickle file and run it on new data set

stacked_loadedmdl = joblib.load('stacked_rf2_svm.pkl')
new_data = pd.DataFrame(data=np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]), columns=['X','Y','Z'])
new_ypred = stacked_loadedmdl.predict(new_data)
print("The predcitions from new data are")
print(new_ypred)

#save to csv
new_df = pd.DataFrame({'Class': [new_ypred]})
with open('new_results.csv', 'w', newline='') as f:
    new_df.to_csv(f)

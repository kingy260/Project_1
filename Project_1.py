import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

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
#initialize logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

#train model
model.fit(X_train, y_train)

#Run the model
y_pred = model.predict(X_test)

#Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

# Step 2: Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

# Step 3: Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

# Step 4: Customize the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

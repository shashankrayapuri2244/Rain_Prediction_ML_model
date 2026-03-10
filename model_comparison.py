import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("weatherAUS.csv")
print("COLUMNS:", df.columns)
print("DATA_SET INFO:")
df.info()

# y -> target , X->features
y = df["RainTomorrow"]
x = df.drop("RainTomorrow", axis = 1)
print("Target Values Sample: " , y.head())
print("Features Shape: " , x.shape)

 # we need to remove the rows where the target is missing if not the model cannot learn if answer itself is missing
print("missing values in target before :",y.isna().sum())

df = df.dropna(subset=["RainTomorrow"])

# Re-create the target after dropping the rows
y = df["RainTomorrow"]
x = df.drop("RainTomorrow", axis =1) 

# missing values in target after 
print("Missing values in target after: ", y.isna().sum())
print("New shape of X: ",x.shape)

# converrting target from yes or no to 0 or 1 since it cannot give strings like yes or no
y = y.map({"No": 0 , "Yes":1})
print("Target values after mapping: ",y.head())

#Handling missing values in x(features)

# identify catogorical and numeric columns
num_cols = x.select_dtypes(include=["int64", "float64"]).columns
cat_cols = x.select_dtypes(include =["object"]).columns

print("The numerical columns are: " ,num_cols)
print("The categorical are: ",cat_cols)

# Need to fill missing values
x[num_cols] = x[num_cols].fillna(x[num_cols].median())
x[cat_cols] = x[cat_cols].fillna(x[cat_cols].mode().iloc[0])

# check remaining missing values 
print("Remaining missing values in x : ",x.isna().sum().sum())

# *** ML models only understands numbers
# so we convert categorical data into numbers using one-hot encoding
# we use pd.get_dummies() for this conversion
x = pd.get_dummies(x, drop_first = True)    
print ("Shape of x after encoding is : ",x.shape)

# Training and testing the model
x_train , x_test , y_train , y_test = train_test_split( x, y, test_size =0.2, random_state =42)
print("Training set shape is: ",x_train.shape)
print("Test set shape is: ",x_test.shape)

# # loading logistic regression
# log_reg = LogisticRegression(max_iter = 100)
# log_reg.fit(x_train, y_train)
# y_pred_lr = log_reg.predict(x_test)
# print("LOGISTIC REGRESSION ACCURACY : ",accuracy_score(y_test, y_pred_lr))

# print("LOGISTIC REGRESSION CONFUSION MATRIX :",confusion_matrix(y_test, y_pred_lr))
# print("LOGISTIC CLASSIFICATION REPORT : ",classification_report(y_test, y_pred_lr))

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

df = DecisionTreeClassifier(random_state = 42)
df.fit(x_train, y_train)
y_pred_dt = df.predict(x_test)
print("PREDICTION : ",df.predict(x_test))
print("\nCONFUSION MATRIX :",confusion_matrix(y_test, y_pred_dt))
print("\nCLASSIFICATION : ",classification_report(y_test , y_pred_dt))
print("\n TEST ACCURACY : ",df.score(x_test ,y_test))
print("\nTRAIN ACCURACY : ",df.score(x_train ,y_train))

# Need to control the decision tree because of overfitting
'''
dt_control =  DecisionTreeClassifier(
    max_depth = 6,
    min_samples_split =100,
    min_samples_leaf = 50,
    class_weight = 'balanced',
    random_state =42
)
dt_control.fit(x_train, y_train)

y_pred_ctrl = dt_control.predict(x_test)

print("\nCONFUSION MATRIX : ",confusion_matrix(y_test , y_pred_ctrl))
print("\nCLASSIFICATION REPORT : ",classification_report(y_test, y_pred_ctrl))
print("\n TEST ACCURACY : ",dt_control.score(x_test ,y_test))
print("\nTRAIN ACCURACY : ",dt_control.score(x_train ,y_train))
'''
# Doing with random forest **previously we predicted the results with decesiontree

rf = RandomForestClassifier(
    n_estimators = 100,
    random_state =42,
    n_jobs =1
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("\nCONFUSION MATRIX : ",confusion_matrix(y_test, y_pred_rf))
print("\nCLASSIFICATION REPORT : ",classification_report(y_test, y_pred_rf))
print("\nTRAIN ACCURACY : ",rf.score(x_train , y_train))
print("\nTEST ACCURACY : ",rf.score(x_test , y_test))

# after rf balanced
rf_balanced = RandomForestClassifier(
    n_estimators = 200,
    class_weight = 'balanced',
    random_state = 42,
    n_jobs = 1
)
rf_balanced.fit(x_train, y_train)

y_pred_rf_bal = rf_balanced.predict(x_test)
print("\nCONFUSION MATRIX : ", confusion_matrix(y_test, y_pred_rf_bal))
print("\nCLASSIFICATION REPORT : ",classification_report(y_test , y_pred_rf_bal))
print("\nTRAIN ACCURACY : ",rf_balanced.score(x_train , y_train))
print("\nTEST ACCURACY : ",rf_balanced.score(x_test , y_test))

# Threshold Tuning
y_prob_rf = rf.predict_proba(x_test)[:, 1]

y_pred_thresh = (y_prob_rf >= 0.35).astype(int)

print("\nCONFUSION MATRIX : ",confusion_matrix(y_test,y_pred_thresh))
print("\n REPORT CLASSIFICATION : ",classification_report(y_test , y_pred_thresh))

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators = 300,
    learning_rate =0.05,
    max_depth = 3,
    random_state = 42
)
gb.fit(x_train , y_train)
y_pred_gb = gb.predict(x_test)   
print("Confusion Matrix: ",confusion_matrix(y_test , y_pred_gb))
print("\nClassification Report: ",classification_report(y_test , y_pred_gb))
print("\nTrain Accuracy: ",gb.score(x_train, y_train))
print("\nTest accuracy: ",gb.score(x_test, y_test))



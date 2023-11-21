#!/usr/bin/env python
# coding: utf-8

# # Importing all required libraries

# In[356]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[357]:


import warnings


# In[358]:


warnings.filterwarnings(action='ignore')


# # Reading the given excel file

# In[359]:


file = pd.ExcelFile('Assignment- Membership woes.xlsx')
file.sheet_names  # see all sheet names


# In[360]:


prob_statement= pd.read_excel("Assignment- Membership woes.xlsx", sheet_name='Problem statement')
prob_statement.head()


# # Exploratory Data Analysis (EDA) & Data Pre-processing:

# In[361]:


data = pd.read_excel("Assignment- Membership woes.xlsx", sheet_name='Data')
data.head()


# In[362]:


data.columns


# In[363]:


data.info()


# In[364]:


data.describe()


# In[365]:


print(len(data))
print(len(data['END_DATE  (YYYYMMDD)'].unique()))
print(len(data['START_DATE (YYYYMMDD)'].unique()))
print(data['MEMBERSHIP_STATUS'].unique()) 
print() 
print(data['MEMBER_GENDER'].unique()) 
print()
print(data['PAYMENT_MODE'].unique()) 
print()
print(data[data['MEMBERSHIP_STATUS']=='INFORCE'].count())


# In[366]:


# Check the proportion of data belonging to each of the classes.
print(data['MEMBERSHIP_STATUS'].value_counts(normalize=True)) 


# In[367]:


# The membership number of a particular user won't we of any use in predicting he's cancelling the subscription or not.
# And then we have already seen that their are too many null values in the END_DATE column for it to be used.

req_data = data.drop(['MEMBERSHIP_NUMBER', 'END_DATE  (YYYYMMDD)', 'AGENT_CODE'],  axis=1)
req_data.head(5)


# In[368]:


print(data.isnull().sum())


# In[369]:


### Nearly ~16% of values in column 'MEMBER_ANNUAL_INCOME' are null, so decided to fill those with mean of column values

print(req_data['MEMBER_ANNUAL_INCOME'].isna().sum())
print(req_data['MEMBER_ANNUAL_INCOME'].mean())


# In[370]:


req_data['MEMBER_ANNUAL_INCOME'].fillna(req_data['MEMBER_ANNUAL_INCOME'].mean(), inplace=True)


# In[371]:


for i in req_data['MEMBER_OCCUPATION_CD'].unique():
    print(i)


# In[372]:


# The null values in the respective columns can be filled with Other/NA etc.

req_data['MEMBER_MARITAL_STATUS'].fillna(value = 'Other', inplace=True)
req_data['MEMBERSHIP_STATUS'].fillna(value = 'Other', inplace=True)
req_data['MEMBERSHIP_PACKAGE'].fillna(value = 'Other', inplace=True)
req_data['PAYMENT_MODE'].fillna(value = 'Other', inplace=True)
req_data['MEMBER_GENDER'].fillna(value = 'Other', inplace=True)
req_data['MEMBER_OCCUPATION_CD'].fillna(value = 5.0, inplace=True)


# In[373]:


# From the date column, we can have the month as it may be a factor
req_data['START_DATE (YYYYMMDD)'] = pd.DatetimeIndex(req_data['START_DATE (YYYYMMDD)']).month


# In[374]:


onehot_req_data = pd.get_dummies(req_data, columns=
     ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE', 'MEMBERSHIP_STATUS'], drop_first=True)
onehot_req_data.head(5)


# In[375]:


# All of the columns in our dataframe, ready to be worked on.
print(onehot_req_data.columns)


# In[376]:


onehot_req_data['PAYMENT_MODE_SINGLE-PREMIUM'].apply(lambda x: type(x) == str)


# In[377]:


import seaborn as sb


# In[378]:


# Plotting the scatter plot and box plot for a few features that we got as important features from the algorithms. 

sb.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_OCCUPATION_CD", data=onehot_req_data)
sb.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_OCCUPATION_CD", data=onehot_req_data)


# In[379]:


sb.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_AGE_AT_ISSUE", data=onehot_req_data)
sb.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_AGE_AT_ISSUE", data=onehot_req_data)


# In[380]:


sb.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBERSHIP_TERM_YEARS", data=onehot_req_data)
sb.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBERSHIP_TERM_YEARS", data=onehot_req_data)


# In[381]:


sb.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_ANNUAL_INCOME", data=onehot_req_data)
sb.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_ANNUAL_INCOME", data=onehot_req_data)


# In[382]:


sb.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="ANNUAL_FEES", data=onehot_req_data)
sb.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="ANNUAL_FEES", data=onehot_req_data)


# # Split the Data into Training and Test Sets:

# In[383]:


#Separating data into two parts, X is independent variables (features) and y is our target variable
data_copy = onehot_req_data.copy()
X = onehot_req_data.drop(['MEMBERSHIP_STATUS_INFORCE'], axis=1)
y = onehot_req_data.drop(['MEMBERSHIP_TERM_YEARS', 'ANNUAL_FEES', 'MEMBER_ANNUAL_INCOME',
                        'MEMBER_OCCUPATION_CD', 'MEMBER_AGE_AT_ISSUE', 'ADDITIONAL_MEMBERS',
                        'START_DATE (YYYYMMDD)', 'MEMBER_MARITAL_STATUS_M',
                        'MEMBER_MARITAL_STATUS_Other', 'MEMBER_MARITAL_STATUS_S',
                        'MEMBER_MARITAL_STATUS_W', 'MEMBER_GENDER_M', 'MEMBER_GENDER_Other',
                        'MEMBERSHIP_PACKAGE_TYPE-B', 'PAYMENT_MODE_MONTHLY',
                        'PAYMENT_MODE_QUARTERLY', 'PAYMENT_MODE_SEMI-ANNUAL',
                        'PAYMENT_MODE_SINGLE-PREMIUM'],axis=1)


# # Feature Selection using selectKBest classifier

# In[384]:


#apply SelectKBest class to extract top 10 best features using chi2 as score_func
bestfeatures = SelectKBest(score_func=chi2, k=5) 
fit = bestfeatures.fit(X,y.to_numpy())
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']
# Printing the feature names and the scores for the top 10 
print(featureScores.nlargest(5,'Score')) 


# # Correlation Matrix

# In[385]:


#get correlations of each features in dataset
corrmat = onehot_req_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sb.heatmap(onehot_req_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Fit the Model using Logistic Regression and finding confusion matrix,accuracy,parameters:

# In[386]:


model = LogisticRegression(C=10**2)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.10, random_state=0, stratify=y)
ytest = ytest.to_numpy() 
model.fit(xtrain, ytrain)
predicted_classes = model.predict(xtest)
accuracy = accuracy_score(ytest.flatten(), predicted_classes)
parameters = model.coef_
print("Accuracy: ", accuracy)
print("\n")
print("Parameters: ", parameters) # printing the coefficients
print("\n")
c_m= confusion_matrix(ytest, predicted_classes) 
print (cm) 


# # Plotting Confusion Matrix

# In[387]:


sb.heatmap(confusion_matrix(ytest, predicted_classes), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Important Features using Random forest

# In[388]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(xtrain, ytrain)


# In[389]:


# True for the features whose importance is greater than the mean importance and False for the rest.
sel.get_support()


# In[390]:


selected_feat=xtrain.columns[(sel.get_support())]
selected_feat


# # churn probability of each user so they can be taken care accordingly.

# In[391]:


data_copy['Churn_probability'] = xg_model.predict_proba(data_copy[xtrain.columns])[:,1]
data_copy.Churn_probability[:5]


# In[396]:


pipe = Pipeline([('classifier' , RandomForestClassifier())])

# Creating parameter grid.
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
    ]
# Create grid search object
gso = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
# Fit on data

best_lrm = gso.fit(xtrain, ytrain)


# In[393]:


y_pred_rf = best_lrm.predict(xtest)


# In[394]:


print(best_lrm.best_params_)
print(classification_report(ytest, y_pred_rf))
print(accuracy_score(ytest.flatten(), y_pred_rf))


# In[395]:


gso1 = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(gso1, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(xtrain, ytrain)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(xtest)

# New Model Evaluation metrics 
print(classification_report(ytest, y_pred_acc))

#Logistic Regression (Grid Search) Confusion matrix
print(confusion_matrix(ytest,y_pred_acc))
print(accuracy_score(ytest.flatten(), y_pred_acc))


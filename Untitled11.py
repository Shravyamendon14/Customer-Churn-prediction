#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[15]:


data = pd.read_excel("customer_churn_large_dataset.xlsx")
data


# In[16]:


new_data = data.drop(columns=['CustomerID', 'Name'])

print(new_data.shape)
new_data.head(3)


# In[17]:


# check for null values
new_data.isna().sum()


# In[18]:


# check for duplicated values
new_data.duplicated().sum()


# In[19]:


# check brief description of data
new_data.describe()


# In[ ]:





# In[22]:


#correlation
# checking for correlation b/w `Age ` and `Monthly_Bill` coln
sns.scatterplot(x=new_data['Monthly_Bill'], y=new_data['Age'], hue=new_data['Churn'])
# checking for correlation b/w `Age ` and `Subscription_Length_Months` coln
sns.scatterplot(x=new_data['Subscription_Length_Months'], y=new_data['Age'], hue=new_data['Churn'])


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate


# In[26]:


#data split
# Divide the data into `features` and `target`
X = new_data.iloc[:, :-1]
y = new_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[27]:


categorical_features = ['Gender', 'Location']
numerical_features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']


one_hot_encoder = OneHotEncoder()
standard_scalar = StandardScaler()


# Create ColumnTransformer Object for `Preprocessing Stuff`
preprocesser = ColumnTransformer(transformers=(
    ('encode_gender', one_hot_encoder, categorical_features),
    ('standardization', standard_scalar, numerical_features)
))


# In[29]:


# Create `Model Pipeline` for Logistic Regression
clf = Pipeline(steps=(
    ('preprocessing', preprocesser),
    ('classifier', LogisticRegression())
))
clf.fit(X_train, y_train)
print("Accuracy score of Logistic Regression is: ", clf.score(X_test, y_test))


# In[30]:


# Check score using other metrics like `Precision Score`, `Recall Score`, `F1 Score`
y_pred = clf.predict(X_test)

print("The precision score of Logistic Regression is: ", precision_score(y_test, y_pred))
print("The recall score of Logistic Regression is: ", recall_score(y_test, y_pred))
print("The F1 score of Logistic Regression is: ", f1_score(y_test, y_pred))


# In[31]:


from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameters to search
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'classifier__penalty': ['l1', 'l2'],  # Regularization type
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model with the best hyperparameters
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy score of the best Logistic Regression model:", accuracy)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# Create a Random Forest Classifier pipeline
rfc_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the Random Forest Classifier to the training data
rfc_pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of Random Forest Classifier:", accuracy)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)







# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameters to search
param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples required at each leaf node
    'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
}

# Create a Random Forest Classifier pipeline
rfc = Pipeline(steps=(
    ('preprocessing', preprocesser),
    ('classifier', RandomForestClassifier(random_state=42))
))

# Create a GridSearchCV object
grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search_rfc.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rfc = grid_search_rfc.best_params_
print("Best Hyperparameters for Random Forest Classifier:", best_params_rfc)

# Evaluate the model with the best hyperparameters
best_model_rfc = grid_search_rfc.best_estimator_
accuracy_rfc = best_model_rfc.score(X_test, y_test)
print("Accuracy score of the best Random Forest Classifier model:", accuracy_rfc)


# In[34]:


get_ipython().system('pip install xgboost')


# In[38]:


from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Define categorical and numerical features
categorical_features = ['Gender', 'Location']
numerical_features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

# Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# Create an XGBoost Classifier pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))  # Use XGBClassifier
])

# Fit the XGBoost Classifier to the training data
xgb_pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = xgb_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of XGBoost Classifier:", accuracy)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:",classification_rep)


# In[36]:


get_ipython().system('pip install xgboost')


# In[41]:


import joblib

# Deploy the model using `pickle` module
import pickle

pickle.dump(clf, open("model.pkl", 'wb'))


# In[ ]:





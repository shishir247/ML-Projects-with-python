#!/usr/bin/env python
# coding: utf-8

# In[303]:


#importing the necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[304]:


#importing the data set
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()


# In[305]:


#printing the description of dataset
print(boston.DESCR)


# In[306]:


#creating dataframe
data = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[307]:


#target variable
(boston.target)[0:10]


# In[308]:


#mutating in dataset
data["TARGET"] = boston.target


# In[309]:


#checking head of data
data.head()


# In[310]:


data.shape


# In[311]:


data.info()


# In[312]:


#checking the null values
data.isnull().sum()


# no column has null values

# In[313]:


#checking description of data
data.describe()


# In[314]:


#plotting pair plot to check relationshiop between all variables
sns.pairplot(data)


# from the above pair plot it can be observed that variable RM and LSTAT shows liner relationship with Target variables

# In[315]:


#plotting distribution plot to check distribution pattern of each variable

rows = 2
cols = 7

fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize=(16,4))

col=data.columns
index = 0

for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]],kde_kws={'bw': 0.1}, ax = ax[i][j])
        index = index +1

plt.tight_layout()


# In[316]:


# seperating dependent and independent variables
X = data.drop(["TARGET"], axis = 1)
y = data.TARGET


# In[317]:


#calculating correlation matrix
corrmat = data.corr()
corrmat


# In[318]:


corrmat.index


# In[319]:


#creating heatmap for correlation matrics data
fig, ax = plt.subplots(figsize = (20,10), facecolor = "w")
sns.heatmap(corrmat, annot=True, annot_kws={"size" : 12})


# In[320]:


#from above heatmap we can conclude that variables RM and LSTAT shows correlation with target variable more than 0.7


# In[321]:


#importing standard scaler and fitting to X
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[322]:


#importing vif and calculating for X scaled
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled


# In[323]:


# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns


# In[324]:


vif


# In[325]:


#dropping the columns RAD as this columns has vif > 5, and is highly correlated with TAX
data.drop(["RAD"], axis=1, inplace=True)


# In[326]:


#Creating a function to check correlation of variables at required thresholds
def getCorrelatedFeature(corrdata,threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index]) > threshold:
            feature.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data = value, index = feature, columns = ["corr value"])
    return df


# In[327]:


#checking variables showing correlation with greater than 0.5
threshold = 0.5
corr_value = getCorrelatedFeature(corrmat["TARGET"], threshold)
corr_value


# In[328]:


#variables 'RM', 'PTRATIO', 'LSTAT' shows correlation with greater than 0.5


# In[329]:


corr_value.index


# In[330]:


#creating correlated data
correlated_data = data[corr_value.index]
correlated_data.head()


# In[331]:


#creating scatter plot for X correlated variables
plt.figure(figsize=(20,30))
plotnumber = 1
for column in X:
    if plotnumber <= 4:
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(X[column], y)
        plt.xlabel(column, fontsize = 20)
        plt.ylabel("Price", fontsize = 20)
    plotnumber = plotnumber + 1
plt.tight_layout()


# In[332]:


#creating pair plot for correlated data
sns.pairplot(correlated_data)
plt.tight_layout()


# In[333]:


#creating heat map for correlated data
fig, ax = plt.subplots(figsize = (5,5), facecolor = "w")
sns.heatmap(correlated_data.corr(), annot=True, annot_kws={"size" : 12})


# In[334]:


correlated_data.head()


# In[335]:


#seperating independent and dependent variables from correlated data
X = correlated_data.drop("TARGET", axis = True)
y = correlated_data.TARGET


# In[336]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[337]:


#splitting correlated data into train and test
X_train, X_test, Y_train, Y_test = train_test_split( X, y , test_size = 0.2, random_state = 24 )


# In[338]:


#fitting the modle
lm.fit(X_train, Y_train)


# In[339]:


#checking y predict values
y_predict = lm.predict(X_test)
y_predict


# In[340]:


#Checking accuracy of modle
lm.score(X_test,y_test)


# In[341]:


#importing R squared
from sklearn.metrics import r2_score


# In[342]:


#checking R squared
score = r2_score(y_test,y_predict)
print("Rsqrd:" , score)


# Regression evaluation metrics:

# In[343]:


from sklearn import metrics


# In[344]:


print("MAE:", metrics.mean_absolute_error(y_test,y_predict))
print("MSE:" , metrics.mean_squared_error(y_test,y_predict))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test,y_predict)))


# In[345]:


#Creating regression plot
rows = 2
cols = 2
fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize=(16,6))

col = correlated_data.columns
index = 0

for i in range(rows):
    for j in range(cols):
        sns.regplot(x = correlated_data[col[index]], y = correlated_data["TARGET"], ax = ax[i][j])
        index = index +1
    fig.tight_layout()


# ### cross validation

# In[346]:


from sklearn.model_selection import cross_validate


# In[347]:


scores = cross_validate(lm,X,y,scoring= "r2",cv = 5, return_train_score= True)


# In[348]:


cv_score = scores["test_score"]
print("5 folds CV scores: {}".format(cv_score))
print("Average cross Validation score for 5 folds: {}".format(np.mean(cv_score)))


# # Alternal method (OLS) orginary least square

# In[349]:


data.head()


# In[350]:


data.columns


# In[351]:


#building model with all the featues
import statsmodels.formula.api as smf
lm1 = smf.ols(formula = 'TARGET ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + TAX + PTRATIO + B + LSTAT'
              ,data = data).fit()


# In[352]:


lm1.rsquared


# In[353]:


lm1.rsquared_adj


# In[354]:


lm1.summary()


# # Applying polynomial regression

# In[355]:


from sklearn.preprocessing import PolynomialFeatures


# In[356]:


degree =2
p_features = PolynomialFeatures(degree=degree)
X_train_poly = p_features.fit_transform(X_train)


# In[357]:


plm = LinearRegression()


# In[358]:


plm.fit(X_train_poly,Y_train)


# In[359]:


y_train_pred = plm.predict(X_train_poly)


# In[360]:


y_test_pred = plm.predict(p_features.fit_transform(X_test))


# In[361]:


#for train model
rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, y_train_pred))
R2_train = metrics.r2_score(Y_train,y_train_pred)


# In[362]:


#for test model
rmse_test = np.sqrt(metrics.mean_squared_error(Y_test,y_test_pred))
R2_test = metrics.r2_score(Y_test,y_test_pred)


# In[370]:


print("Model performance for Training set at degree-2 is :")
print("-"*50)
print("rmse of training set is : ", rmse_train)
print("Rsqrd for training set is :", R2_train)
print("\n")
print("Model performance for Testing set at degree-2 is :")
print("-"*50)
print("rmse of test set is :", rmse_test)
print("Rsqrd for test set is :", R2_test)


# #### lets try a few degrees 3 to 20 and observe the RMSE and R2¶
# -wrap the above in a function

# In[364]:


def poly_function(degrees):
    
    degree_list      = []
    train_rmse_list  = []
    train_r2_list    = []
    test_rmse_list   = []
    test_r2_list     = []
    
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)

        # transforms the existing features to higher degree features.
        X_train_poly = poly_features.fit_transform(X_train)

        # fit the transformed features to Linear Regression
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, Y_train)

        # predicting on training data-set
        y_train_pred = poly_model.predict(X_train_poly)

        # predicting on test data-set
        y_test_pred = poly_model.predict(poly_features.fit_transform(X_test))

        # evaluating the model on training dataset
        rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, y_train_pred))
        r2_train   = metrics.r2_score(Y_train, y_train_pred)

        # evaluating the model on test dataset
        rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, y_test_pred))
        r2_test   = metrics.r2_score(Y_test, y_test_pred)
    
        degree_list.append(degree)
        train_rmse_list.append(rmse_train)
        train_r2_list.append(r2_train)
        test_rmse_list.append(rmse_test)
        test_r2_list.append(r2_test)
        
    return degree_list, train_rmse_list, train_r2_list, test_rmse_list, test_r2_list
    


# In[365]:


degree_list = np.arange(1, 20, 1)

degree_list, train_rmse_list, train_r2_list, test_rmse_list, test_r2_list = poly_function(degree_list)

results= pd.DataFrame({'degrees':   np.array(degree_list), 
                       'train_mse': np.array(train_rmse_list),
                       'train_r2':  np.array(train_r2_list), 
                       'test_mse':  np.array(test_rmse_list),
                       'test_r2':   np.array(test_r2_list)
                      })


# In[366]:


results


# In[367]:


plt.figure(figsize=(12, 6))

# Train MSE
plt.plot(results.degrees, results.train_mse, color='r', label='Train RMSE', alpha=1)

# Test MSE
plt.plot(results.degrees, results.test_mse, color='g', label='test RMSE', alpha=1)

plt.legend();


# In[368]:


results = results[results.test_r2>=0]


# In[369]:


plt.figure(figsize=(12, 6))

# Train R2
plt.plot(results.degrees, results.train_r2, color='r', label='Train R2', alpha=1)

# Test R2
plt.plot(results.degrees, results.test_r2, color='g', label='test R2', alpha=1)

plt.legend();


# over fitting for test started after degree 2 so we will choose degree-2

# In[ ]:





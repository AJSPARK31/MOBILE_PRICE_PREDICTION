#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing important libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,KFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[28]:


# importing the data
data=pd.read_csv('mobile_price_data_new.csv')


# In[29]:


print(data)


# In[30]:


print(data.isnull().sum())


# In[31]:


data.info()


# In[33]:


#unique values in the columns

data.nunique()

# bluetooth column has only one unique value so we can drop the bluetooth column


# In[39]:


# dropping the bluetooth column
print(data.shape)
data=data.drop(['bluetooth'],axis=1)
print(data.shape)


# In[40]:


print(data)


# In[44]:


#visualize the data

plt.bar(data['mobile_name'],data['mobile_price'])
plt.show()


# In[49]:


plt.scatter(data['mobile_color'],data['mobile_price'])
plt.show()


# In[48]:


import seaborn as sns
sns.heatmap(data.corr(),annot=True)


# In[51]:


plt.scatter(data['dual_sim'],data['mobile_price'])
plt.show()
# here we can see that single sim has only one data point,
# hence it will not be good attribute for predicting the mobile price 
# so we an frop the dual_sim column
data=data.drop(['dual_sim'],axis=1)


# In[52]:


plt.scatter(data['disp_size'],data['mobile_price'])
plt.show()


# In[54]:


plt.scatter(data['os'],data['mobile_price'])
plt.show()


# In[55]:


plt.scatter(data['num_cores'],data['mobile_price'])
plt.show()


# In[56]:


plt.scatter(data['mp_speed'],data['mobile_price'])
plt.show()


# In[57]:


plt.scatter(data['int_memory'],data['mobile_price'])
plt.show()


# In[58]:


plt.scatter(data['p_cam'],data['mobile_price'])
plt.show()


# In[59]:


plt.scatter(data['f_cam'],data['mobile_price'])
plt.show()


# In[60]:


plt.scatter(data['network'],data['mobile_price'])
plt.show()


# In[61]:


plt.scatter(data['battery_power'],data['mobile_price'])
plt.show()


# In[62]:


plt.scatter(data['mob_width'],data['mobile_price'])
plt.show()


# In[64]:


plt.scatter(data['mob_height'],data['mobile_price'])
plt.show()


# In[65]:


plt.scatter(data['mob_depth'],data['mobile_price'])
plt.show()


# In[66]:


plt.scatter(data['mob_weight'],data['mobile_price'])
plt.show()


# In[67]:


# splitting the data into features and target 

X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[68]:


print(X)


# In[69]:


print(y)


# In[70]:


# splitting the data into train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=7)


# In[72]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[73]:


# splitting the data into categorical and numerical 

X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include=['int32','int64','float32','float64'])


# In[74]:


print(X_train_cat)


# In[76]:


print(X_train_num)


# In[77]:


X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include=['int32','int64','float32','float64'])


# In[78]:


print(X_test_cat)


# In[79]:


print(X_test_num)


# In[82]:


# performing the preprocessing techniques for train and test data

oe=OrdinalEncoder()
oe.fit(X_train_cat)
X_train_cat_enc=oe.transform(X_train_cat)

oe.fit(X_test_cat)
X_test_cat_enc=oe.transform(X_test_cat)


# In[85]:


print(X_train_cat_enc)
print(X_test_cat_enc)


# In[87]:


ss=StandardScaler()
ss.fit(X_train_num)
X_train_num_enc=ss.transform(X_train_num)

ss.fit(X_test_num)
X_test_num_enc=ss.transform(X_test_num)


# In[89]:


# preprocessing for target
y_train_df=pd.DataFrame(y_train)
y_test_df=pd.DataFrame(y_test)


ss.fit(y_train_df)
y_train_enc=ss.transform(y_train_df)


ss.fit(y_test_df)
y_test_enc=ss.transform(y_test_df)


# In[90]:


# concating the categorical and numerical data  for train test data
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)
X_train_num_enc_df=pd.DataFrame(X_train_num_enc)
X_test_num_enc_df=pd.DataFrame(X_test_num_enc)



# In[97]:


X_train_final=pd.concat([X_train_cat_enc_df,X_train_num_enc_df],axis=1)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num_enc_df],axis=1)


# In[98]:


# model building


lr=LinearRegression()
lr.fit(X_train_final,y_train_enc)
y_pred=lr.predict(X_test_final)
MSE=mean_squared_error(y_pred,y_test_enc)
print(MSE)


# In[99]:


from sklearn.linear_model import Ridge , Lasso


# In[114]:


RR=Ridge(alpha=.001, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
RR.fit(X_train_final,y_train_enc)
y_pred_r=RR.predict(X_test_final)
MSE_R=mean_squared_error(y_pred_r,y_test_enc)
print(MSE_R)


# In[119]:


LR=Lasso(alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
LR.fit(X_train_final,y_train_enc)
y_pred_l=LR.predict(X_test_final)

MAE_L=mean_absolute_error(y_pred_l,y_test_enc)
print(MAE_L)


# In[124]:


from sklearn.tree import DecisionTreeRegressor
DTR=DecisionTreeRegressor()
DTR.fit(X_train_final,y_train_enc)
y_pred_dtr=DTR.predict(X_test_final)
MSE_dtr=mean_squared_error(y_pred_dtr,y_test_enc)
print(MSE_dtr)


# In[ ]:





# In[ ]:





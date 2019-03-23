
# coding: utf-8

# ## Create Logistic regression to predict Absenteeism

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data_preprocessed = pd.read_csv('Absenteeism-preprocessed.csv')


# In[4]:


data_preprocessed.head(5)


# In[5]:


data_preprocessed['Absenteeism Time in Hours'].median()


# In[7]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > data_preprocessed['Absenteeism Time in Hours'].median(), 1,0 )


# In[8]:


data_preprocessed['Extensive Absenteeism'] = targets


# In[9]:


data_preprocessed.head(5)


# In[10]:


data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'],axis=1)


# In[11]:


data_with_targets is data_preprocessed


# In[12]:


data_with_targets.shape


# In[16]:


unscaled_inputs = data_with_targets.iloc[:,:-1]


# In[17]:


from sklearn.preprocessing import StandardScaler
absenteeism_scaler = StandardScaler()


# In[101]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self, columns,copy=True, with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None,copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    


# In[102]:


unscaled_inputs.columns.values


# In[103]:


columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
       'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pet']


# In[104]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[105]:


absenteeism_scaler.fit(unscaled_inputs)


# In[106]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[107]:


scaled_inputs


# In[108]:


scaled_inputs.shape


# In[109]:


from sklearn.model_selection import train_test_split


# In[110]:


x_train , x_test, y_train, y_test = train_test_split(scaled_inputs,targets,train_size=0.8, random_state=20)


# In[111]:


print(x_train.shape, y_train.shape)


# In[112]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[113]:


reg = LogisticRegression()
reg.fit(x_train, y_train)


# In[114]:


reg.score(x_train,y_train)


# In[115]:


model_outputs = reg.predict(x_train)


# In[116]:


model_outputs == y_train


# In[117]:


np.sum(model_outputs == y_train)


# In[118]:


model_outputs.shape[0]


# In[119]:


np.sum((model_outputs==y_train))/model_outputs.shape[0]


# In[120]:


reg.intercept_


# In[121]:


reg.coef_


# In[122]:


feature_name = unscaled_inputs.columns.values


# In[123]:


summary_table = pd.DataFrame( columns=['Feature Name'], data = feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table


# In[124]:


summary_table.index = summary_table.index + 1
summary_table.loc[0]= ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# In[125]:


summary_table['Odds_ration'] = np.exp(summary_table.Coefficient)


# In[126]:


summary_table


# In[127]:


summary_table.sort_values('Odds_ration',ascending=False)


# In[128]:


reg.score(x_test, y_test)


# In[129]:


predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[78]:


predicted_proba.shape


# In[81]:


predicted_proba[:,1]


# ## Save the model

# In[82]:


import pickle


# In[130]:


with open('model', 'wb') as file :
    pickle.dump(reg,file)


# In[131]:


with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler,file)



# coding: utf-8

# In[1]:


import pandas as pd


# In[37]:


raw_csv_data = pd.read_csv('Absenteeism-data.csv')


# In[38]:


raw_csv_data.head(10)


# In[39]:


df = raw_csv_data.copy()


# In[40]:


df.head(10)


# In[41]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[42]:


display(df)


# In[43]:


df.info()


# In[44]:


df = df.drop(['ID'],axis=1)


# In[45]:


df


# In[50]:


len(df['Reason for Absence'].unique())


# In[51]:


reason_column = pd.get_dummies(df['Reason for Absence'])


# In[52]:


reason_column


# In[53]:


reason_column['check'] = reason_column.sum(axis=1)


# In[54]:


reason_column


# In[56]:


reason_column['check'].sum(axis=0)


# In[57]:


reason_column['check'].unique()


# In[58]:


reason_column = reason_column.drop(['check'],axis=1)


# In[59]:


reason_column


# In[60]:


reason_column = pd.get_dummies(df['Reason for Absence'],drop_first=True)


# In[61]:


reason_column


# In[65]:


reason_type_1 = reason_column.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_column.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_column.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_column.loc[:,22:].max(axis=1)


# In[68]:


df = df.drop(['Reason for Absence'],axis=1)


# In[71]:


df = pd.concat([df,reason_type_1,reason_type_2,reason_type_3,reason_type_4],axis=1)


# In[72]:


df


# In[73]:


df.columns.values


# In[74]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1','Reason_2', 'Reason_3','Reason_4']


# In[75]:


df.columns = column_names


# In[76]:


df.head(5)


# In[77]:


column_name_reordered = ['Reason_1','Reason_2', 'Reason_3','Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[78]:


df = df[column_name_reordered]
df.head(5)


# In[162]:


df_reason_mod = df.copy()


# In[163]:


df_reason_mod


# In[161]:


df_reason_mod['Date']=pd.to_datetime(df_reason_mod['Date'])


# In[84]:


df_reason_mod['Date']


# In[164]:


df_reason_mod['Date']=pd.to_datetime(df_reason_mod['Date'],format = '%d/%m/%Y')


# In[165]:


df_reason_mod['Date']


# In[166]:


df_reason_mod.info()


# In[167]:


df_reason_mod['Date'][0].month
list_month = []


# In[168]:


for i in range(df_reason_mod.shape[0]):
    list_month.append(df_reason_mod['Date'][i].month)


# In[169]:


df_reason_mod['Month Value'] = list_month


# In[170]:


df_reason_mod.head(5)


# In[171]:


df_reason_mod['Date'][699].weekday()


# In[172]:


def date_to_weekday(date_value):
    return date_value.weekday()


# In[173]:


df_reason_mod['Day of the Week']=df_reason_mod['Date'].apply(date_to_weekday)


# In[174]:


df_reason_mod.head(5)


# In[175]:


df_reason_mod = df_reason_mod.drop(['Date'],axis=1)


# In[176]:


df_reason_mod


# In[177]:


df_reason_mod.columns.values


# In[141]:


column_re = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value',
       'Day of the Week','Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours' ]


# In[178]:


df_reason_mod = df_reason_mod[column_re]


# In[179]:


df_reason_mod.head(5)


# In[183]:


df_reason_mod['Education'].value_counts()


# In[184]:


df_reason_mod['Education'] = df_reason_mod['Education'].map({1:0, 2:1, 3:1, 4:1})


# In[186]:


df_reason_mod['Education'].unique()


# In[187]:


df_reason_mod['Education'].value_counts()


# In[188]:


df_preprocessed = df_reason_mod.copy()
df_preprocessed.head(5)


#!/usr/bin/env python
# coding: utf-8

# In[197]:


import pandas as pd


# In[198]:


data=pd.read_csv('train.csv')


# In[199]:


data


# ### 1. Display top 10 rows of the data set

# In[200]:


data.head(10)


# ### 2. Display bottom 10 rows of the data set

# In[201]:


data.tail(10)


# ### 3. Find the shape of the dataset.

# In[202]:


data.shape


# In[203]:


print('Number of Rows = ',data.shape[0])
print('Number of Columns = ',data.shape[1])


# ### 4. Get Information of the dataset.

# In[204]:


data.info()


# ### 5. Get overall statistics of the dataset.

# In[205]:


data.describe(include='all')


# ### 6. Find the Null values.

# In[206]:


data.isnull().sum()


# In[207]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[208]:


sns.heatmap(data.isnull())


# In[209]:


percentage_missing=data.isnull().sum()*100/len(data)
percentage_missing


# In[210]:


len(data)


# ### 7. Data filtering

# In[211]:


data.columns


# In[212]:


data[data['Sex']=='male']


# In[213]:


sum(data['Sex']=='male')


# In[214]:


data[['Name','Age']]


# In[215]:


sum(data['Survived']==1)


# ### 8. Drop the columns with higher missing values

# In[216]:


data.drop('Cabin',inplace=True,axis=1)


# In[217]:


data


# In[218]:


data.isnull().sum()


# ### 9. Handling missing values

# In[219]:


data['Embarked'].mode()


# In[220]:


data['Embarked'].fillna('S',inplace=True)


# In[221]:


data.isnull().sum()


# In[222]:


data['Age'].mean()


# In[223]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[224]:


data.isnull().sum()


# ### 10. Categorical data encoding

# In[225]:


data.head()


# In[226]:


data['Sex'].unique()


# In[227]:


data['Gender']=data['Sex'].map({'male':1,'female':0})


# In[228]:


data


# In[229]:


x=data['Sex'].map({'male':1,'female':0})
data.insert(5,'Gender_New',x)


# In[230]:


data


# In[231]:


data['Embarked'].unique()


# In[232]:


data_dummies=pd.get_dummies(data,columns=['Embarked'])


# In[233]:


data_dummies.columns


# In[234]:


data1=pd.get_dummies(data,columns=['Embarked'],drop_first=True)


# In[235]:


data1


# ### 11. Univariate Analysis

# In[236]:


data.columns


# #### A. How many people survived and how many died?

# In[237]:


data['Survived'].value_counts()


# In[243]:


sns.countplot(x='Survived',data=data)
plt.title('Survived v/s Died')
plt.show()


# #### B. How many passangers were in First Class(FC), Second Class(SC) and Third Class(TC)?

# In[245]:


data.columns


# In[246]:


data['Pclass'].value_counts()


# In[251]:


sns.countplot(x='Pclass',data=data)


# #### C. Number of Male and Female passangers.

# In[253]:


data.columns


# In[255]:


data['Sex'].value_counts()


# In[256]:


sns.countplot(x='Sex', data=data)


# In[263]:


plt.hist(data['Age'])
plt.show()


# In[264]:


sns.boxplot(x='Age',data=data)
plt.show()


# ### 12. Bivariate Analysis

# #### A. Who has better chance of survival Male or Female?

# In[266]:


data.columns


# In[268]:


sns.barplot(x='Sex',y='Survived',data=data)
plt.show()


# #### B. Which passenger class has better chance of survival FC, SC or TC?

# In[270]:


sns.barplot(x='Pclass',y='Survived',data=data)
plt.show()


# ### 13. Feature Engineering

# In[272]:


data.columns


# In[274]:


data['Family_Size']=data['SibSp']+data['Parch']


# In[275]:


data


# In[276]:


data.columns


# In[277]:


data['Fare_per_person']=data['Fare']/(data['Family_Size']+1)


# In[278]:


data['Fare_per_person']


# In[ ]:





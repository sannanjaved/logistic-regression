#%%
import matplotlib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.rcmod import set_style 
train = pd.read_csv ('/home/sannan/sannan/tutorial practice/logistic regression/titanic_train.csv')



#%%

sns.heatmap(train.isnull(),yticklabels =False,cbar=False,cmap='viridis')
train.head()
plt.show()

# %%

sns.set_style('whitegrid')
sns.countplot(x= 'Survived',hue ='Pclass' ,data =train, )
plt.show()





# %%


# 


# # %%

# import cufflinks as cf
# cf.go_offline()
# plt.figure(figsize=(10,7))
# sns.boxplot(x ='Pclass', y='Age',data= train)
# # %%


# # %%
# def impute_age (cols):
#     Age =cols[0]
#     Pclass =cols[1]

#     if pd.isnull(Age):
#         return 37
#     elif Pclass ==2:
#             return 29
#     else:
#         return 24
        
# %%

train.head()

# %%
train.drop('Cabin',axis =1 , inplace=True)
# %%
train.head()
sns.heatmap(train.isnull(),yticklabels =False,cbar=False,cmap='viridis')
# %%
pd.get_dummies(train['Sex'],drop_first=True)
# %%

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace =True)
train.head()

# %%


train.drop('PassengerId',axis=1, inplace=True)
train.head()
# %%
x= train.drop('Survived',axis=1)
y = train['Survived']
# %%
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(  x, y, test_size=0.33, random_state=101)


# %%
from sklearn.linear_model import LogisticRegression

# %%
logmodel =LogisticRegression()
# %%

logmo
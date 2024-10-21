#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Youtube01-Psy.csv')


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.head()
df.tail(10)


# In[6]:


df.dtypes[df.dtypes=='object']
columns = df.columns
columns
df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df = df[["CONTENT", "CLASS"]]


# In[9]:


df


# In[10]:


df.hist(figsize=(15,15), xrot=-45, bins=10) ## Display the labels rotated by 45 degress


# In[11]:


# Clear the text "residue"
plt.show()
plt.style.use('bmh')
sns.boxplot(x=df['CLASS'], y=df['CONTENT'])
plt.title('youtube spam', fontdict = {'fontname' : 'Monospace', 'fontsize' : 15, 'fontweight' : 
'bold'})
plt.xlabel('CLASS', fontdict = {'fontname' : 'Monospace', 'fontsize' : 10})
plt.ylabel('CONTENT', fontdict = {'fontname' : 'Monospace', 'fontsize' : 10})
plt.tick_params(labelsize = 7)
plt.show()


# In[12]:


plt.style.use('ggplot')
plt.figure(figsize = (20, 7))
sns.distplot(df['CLASS'], bins = 20)
plt.title('youtube spam', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 
'bold'})
plt.xlabel('CLASS', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})
plt.ylabel('spam', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})
plt.tick_params(labelsize = 15)
plt.show()




# In[14]:


X = df["CONTENT"]
y = df["CLASS"]
X
y
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[15]:


X_scaled = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.naive_bayes import BernoulliNB


# In[16]:


print(X_train)


# In[17]:


print(y_train)


# In[18]:


print(y_test)


# In[19]:


nb = BernoulliNB()
nb.fit(X_train, y_train)
acc=nb.score(X_test, y_test)
acc
sample = "Follow me: https://example.com"
test = cv.transform([sample]).toarray()
nb.predict(test)


# In[20]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)


# In[21]:


Y_pred_lin_reg=lin_reg.predict(X_test)
Acc1=lin_reg.score(X_test, y_test)
from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor()


# In[22]:


dec_tree.fit(X_train, y_train)
Y_pred_dec = dec_tree.predict(X_test)
Acc2=dec_tree.score(X_test, y_test)
Acc2
from sklearn.ensemble import RandomForestRegressor
ran_for = RandomForestRegressor()


# In[23]:


ran_for.fit(X_train, y_train)
Y_pred_ran=ran_for.predict(X_test)
Acc3=ran_for.score(X_test, y_test)
Acc3
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Linear Regression: ")
print("RMSE:",np.sqrt(mean_squared_error(y_test, Y_pred_lin_reg)))
print("R2 score:", r2_score(y_test, Y_pred_lin_reg))
print("Decision tree regression: ")
print("RMSE:",np.sqrt(mean_squared_error(y_test, Y_pred_dec)))


# In[24]:


print("R2 score:", r2_score(y_test, Y_pred_dec))
print("Random forest regression: ")
print("RMSE:",np.sqrt(mean_squared_error(y_test, Y_pred_ran)))
print("R2 score:", r2_score(y_test, Y_pred_ran))


# In[25]:


from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
nb.fit(X_train, y_train)
acc=nb.score(X_test, y_test)
acc


# In[26]:


import matplotlib.pyplot as plt; plt.rcdefaults()
objects = ('nb',' LR','DT','RF')
y_pos = np.arange(len(objects))
performance = [acc,Acc1,Acc2,Acc3]
plt.bar(y_pos, performance, align='center', alpha=0.15)
plt.xticks(y_pos,objects)
plt.ylabel('Accuracy')
plt.title('Linear regressin vs Decision tree vs Random forest vs naive_bayes vs svm')
plt.show()

# In[28]:


import joblib


# Save the trained model
joblib.dump(cv,'vectorizer.pkl')
joblib.dump(nb,'spam_detector_model.pkl')














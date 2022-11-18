#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install missingno')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import missingno as msn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[11]:


df=pd.read_csv('chronickidneydisease.csv')


# In[12]:


df.shape


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:


df.info()


# In[16]:


df.dtypes


# In[17]:


df.isnull().sum()


# In[18]:


f, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(df.isnull(),yticklabels=False,cmap="crest")


# In[19]:


df["age"]=df["age"].fillna(df["age"].mean())


# In[20]:


df['age'].isnull().sum()


# In[21]:


df.columns


# In[22]:


df["sg"]=df["sg"].fillna(df["sg"].mean())
df["al"]=df["al"].fillna(df["al"].mean())
df["bp"]=df["bp"].fillna(df["bp"].mean())


# In[23]:


numerical=[]
for col in df.columns:
    if df[col].dtype=="float64":
        numerical.append(col)
print(numerical)
for col in df.columns:
    if col in numerical:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


df.corr()


# In[27]:


plt.figure(figsize=(15,8));
plt.title("Correlation",color="blue")
sns.heatmap(df.corr(),linewidth=1,annot=True);


# In[29]:


df.duplicated().value_counts() #checking for repetation


# In[30]:


df['classification'].value_counts()


# In[31]:


df["classification"]=df["classification"].replace("ckd\t","ckd",regex=True)


# In[32]:


df['classification'].value_counts()


# In[33]:


df.drop('id',axis=1,inplace=True)


# In[34]:


df.head()


# In[35]:


df.dtypes


# In[36]:


sns.set_theme(style="darkgrid")
fig, ((ax1, ax2,ax3,ax4,ax5), (ax6, ax7,ax8,ax9,ax10))= plt.subplots(nrows=2, ncols=5, figsize=(18,14))
sns.boxplot(data=df,x="age",ax=ax1)
sns.boxplot(data=df,x="bp",ax=ax2)
sns.boxplot(data=df,x="sg",ax=ax3)
sns.boxplot(data=df,x="al",ax=ax4)
sns.boxplot(data=df,x="bgr",ax=ax5)
sns.boxplot(data=df,x="bu",ax=ax6)
sns.boxplot(data=df,x="sc",ax=ax7)
sns.boxplot(data=df,x="sod",ax=ax8)
sns.boxplot(data=df,x="pot",ax=ax9)
sns.boxplot(data=df,x="hemo",ax=ax10)


# In[37]:


p25 = df['bgr'].quantile(0.25)
p75 = df['bgr'].quantile(0.75)
iqr=p75-p25
# Finding upper and lower limit
upper_limit = p75 + 1.5 * iqr
lower_limit = p25 - 1.5 * iqr
df[df['bgr'] > upper_limit]
df[df['bgr'] < lower_limit]
#Trimming the outlier
new_df = df[df['bgr'] < upper_limit]


# In[38]:


p25 = df['sc'].quantile(0.25)
p75 = df['sc'].quantile(0.75)
iqr=p75-p25
# Finding upper and lower limit
upper_limit = p75 + 1.5 * iqr
lower_limit = p25 - 1.5 * iqr
df[df['sc'] > upper_limit]
df[df['sc'] < lower_limit]
#Trimming the outlier
new_df = df[df['sc'] < upper_limit]


# In[39]:


p25 = df['bu'].quantile(0.25)
p75 = df['bu'].quantile(0.75)
iqr=p75-p25
# Finding upper and lower limit
upper_limit = p75 + 1.5 * iqr
lower_limit = p25 - 1.5 * iqr
df[df['bu'] > upper_limit]
df[df['bu'] < lower_limit]
#Trimming the outlier
new_df = df[df['bu'] < upper_limit]


# In[40]:


df.dtypes


# In[41]:


fig, ax = plt.subplots(figsize=(16,12), ncols=3, nrows=3)
sns.set_style("dark")
sns.set_context("notebook")

sns.distplot(df['age'],kde =True, ax=ax[0][0])
sns.distplot(df['bp'],   kde =True, ax=ax[0][1])
sns.distplot(df['sg'],  kde =True, ax=ax[0][2])
sns.distplot(df['al'],  kde =True, ax=ax[1][0])
sns.distplot(df['su'],  kde =True, ax=ax[1][1])
sns.distplot(df['bgr'],  kde =True, ax=ax[1][2])
sns.distplot(df['bu'],  kde =True, ax=ax[2][0])
sns.distplot(df['sc'],  kde =True, ax=ax[2][1])
sns.distplot(df['sod'],  kde =True, ax=ax[2][2])


# In[42]:


df.dtypes


# In[43]:


plt.figure(figsize=(8,8))
sns.scatterplot(data=df, x="age", y="sg", hue="classification",palette = "crest")


# In[44]:


sns.catplot(data=df, x="ba", y="age",palette = "crest")


# In[45]:


sns.set_style('darkgrid')
ax = sns.boxplot(x='classification',y='bu', hue = 'classification', data=df,width=0.8, dodge=False)
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['CKD','No CKD'], bbox_to_anchor=(1.35,1),
                         title = 'Dianostic Classification')
ax.set_title(' Blood Urea vs Chronic Kidney Disease(CKD)',fontsize=17)
ax.set_xlabel('Diagnosed with CKD',fontsize=15)
ax.set_ylabel('Blood Urea Levels (mg/dL)',fontsize=15)
plt.show()


# In[46]:


sns.set_style('darkgrid')
ax = sns.boxplot(x='classification',y='al', hue = 'classification', data=df,width=0.8, dodge=False)
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['CKD','No CKD'], bbox_to_anchor=(1.35,1),
                         title = 'Dianostic Classification')
ax.set_title('albumin vs Chronic Kidney Disease(CKD)',fontsize=17)
ax.set_xlabel('Diagnosed with CKD',fontsize=15)
ax.set_ylabel('Serum Creatinine (mg/dL)',fontsize=15)
plt.show()


# In[47]:


le = LabelEncoder()
object_col = [col for col in df.columns if df[col].dtype == 'object']
for col in object_col:
    df[col] = le.fit_transform(df[col])


# In[48]:


df.dtypes


# In[49]:


X=df[[ 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']]
y=df[['classification']]


# In[50]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[51]:


print("Training Data ::-")
print("The shape of X training data is :-" ,X_train.shape)
print("The shape of y training data is :-" ,y_train.shape)


# In[52]:


print("Testing Data ::-")
print("The shape of X testing data is :-" ,X_test.shape)
print("The shape of y testing data is :-" ,y_test.shape)


# In[53]:


ss=StandardScaler()
X=ss.fit_transform(X)
X.shape


# In[54]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=222)


# In[55]:


model=LogisticRegression(max_iter=200,random_state=0)
model


# In[56]:


model.fit(X_train,y_train)


# In[57]:


y_predic=model.predict(X_test)
print(y_predic)


# In[58]:


print("Accuracy of the model is :  %3f " % accuracy_score(y_test,y_predic))


# In[59]:


print(classification_report(y_test, y_predic))


# In[60]:


model=DecisionTreeClassifier(random_state=15)
model.fit(X_train,y_train)


# In[61]:


y_predict=model.predict(X_test)
print(y_predict)


# In[62]:


print("Accuracy of the model is :  %3f " % accuracy_score(y_test,y_predic))


# In[63]:


print(classification_report(y_test, y_predict))


# In[64]:


from sklearn.ensemble import RandomForestClassifier
# create Classifier object
classi = RandomForestClassifier(n_estimators = 500, random_state = 0)


# In[65]:


classi.fit(X_train, y_train)


# In[66]:


classi.score(X_train,y_train)*100


# In[67]:


classi.fit(X_test, y_test)
classi.score(X_test,y_test)*100


# In[68]:


model=KNeighborsClassifier()
model.fit(X_train,y_train)


# In[69]:


y_predict=model.predict(X_test)
print(y_predict)


# In[70]:


print(classification_report(y_test, y_predict))


# In[71]:


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[72]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





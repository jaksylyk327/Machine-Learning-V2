#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
print('Done')


# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')
print('Done')


# In[7]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[8]:


df.shape


# In[9]:


df.columns


# In[3]:


get_ipython().system('pip install seaborn')


# In[4]:


get_ipython().system('conda install -c anaconda seaborn -y')


# In[10]:


df['loan_status'].value_counts()


# In[11]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[13]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[14]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[15]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[16]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[17]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[18]:


df['deadline']=df['due_date']-df['effective_date']
df.head()


# In[19]:


df['deadline']=df['deadline'].dt.days


# In[20]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[21]:


df[['Principal','terms','age','Gender','education']].head()


# In[22]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[23]:


X = Feature
X[0:5]


# In[24]:


y = df['loan_status'].values
y[0:5]


# In[25]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ## KNN

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k=15
mean_acc = np.zeros((k-1))

for n in range(1,k):
    
    Knn= KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=Knn.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    
mean_acc


# In[28]:


import matplotlib.pyplot as plt
plt.plot(range(1,k),mean_acc)
plt.xlabel("Number of Neighbor(K)")
plt.ylabel("Accuracy")
plt.show()


# In[30]:


print('The best accuracy is',mean_acc.max(),'with K=',mean_acc.argmax()+1)


# In[31]:


k=7

f_knn=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
f_yhat=f_knn.predict(X_test)
print('Trainset-accuracy', metrics.accuracy_score(y_train,f_knn.predict(X_train)))
print('Testset-accuracy', metrics.accuracy_score(y_test,f_yhat))


# In[32]:


from sklearn.metrics import jaccard_score 
from sklearn.metrics import f1_score

accu_matrix={}

accu_matrix['knn_jaccard']=jaccard_score(y_test, f_yhat,pos_label = "PAIDOFF")
accu_matrix['knn_f1_score']=f1_score(y_test,f_yhat, average='weighted')
accu_matrix


# ## Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


k=15
mean_acc = np.zeros((k-1))

for n in range(1,k):
    
    test_tree= DecisionTreeClassifier(criterion="entropy",max_depth=n).fit(X_train,y_train)
    yhat=test_tree.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    
mean_acc


# In[35]:


plt.plot(range(1,k),mean_acc)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()
print('The best accuracy is',mean_acc.max(),'when max_depth=',mean_acc.argmax()+1)


# In[36]:


dec_tree= DecisionTreeClassifier(criterion="entropy",max_depth=6)
dec_tree.fit(X_train,y_train)

yhat= dec_tree.predict(X_test)

# Lets print the accuracy 
print('trainset accuracy', metrics.accuracy_score(y_train, dec_tree.predict(X_train)))
print('testset accuracy', metrics.accuracy_score(y_test, yhat))


# In[37]:


get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydotplus')
import graphviz 
import pydotplus


# In[38]:


from six import StringIO
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "Decision_Tree.png"
featureNames = Feature.columns
out=tree.export_graphviz(dec_tree,feature_names=featureNames, out_file= dot_data,class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# ## SVM

# In[39]:


from sklearn.svm import SVC

svm= SVC(kernel='linear').fit(X_train,y_train)
yhat= svm.predict(X_test)


# In[40]:


print('trainset accuracy', metrics.accuracy_score(y_train, svm.predict(X_train)))
print('testset accuracy', metrics.accuracy_score(y_test, yhat))


# ## Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


l_reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
yhat=l_reg.predict(X_test)
yprob= l_reg.predict_proba(X_test)


# In[43]:


from sklearn.metrics import log_loss

accu_matrix['l_reg_jaccard']= jaccard_score(y_test, f_yhat,pos_label = "PAIDOFF")
accu_matrix['l_reg_f1_score']= f1_score(y_test,yhat, average='weighted')
accu_matrix['l_reg_logloss']= log_loss(y_test, yprob)

accu_matrix


# ## Model Evaluation using Test Set

# In[44]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
print('Done')


# In[45]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[57]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[58]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()


# In[59]:


test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
Feature_test = test_df[['Principal','terms','age','Gender','weekend']]
Feature_test.head()


# In[60]:


Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
test_X = Feature_test
test_X[0:5]
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]
test_y = test_df['loan_status'].values
test_y[0:5]


# In[61]:


test_X


# # KNN test Data Evaluation metrics- Test Data

# In[62]:


from sklearn import metrics
yhat_test = f_knn.predict(test_X)

knn_jaccard = jaccard_score(test_y, yhat_test,pos_label = "PAIDOFF")
print("Jaccard Accuracy : ",knn_jaccard)
knn_f1 = f1_score(test_y, yhat_test, average='weighted')
print("F1 Score : ",knn_f1)


# # Decision Tree Evaluation -Test Data

# In[63]:


Pred_dtree = dec_tree.predict(test_X)

dtree_jaccard = jaccard_score(test_y, Pred_dtree,pos_label = "PAIDOFF")
print("Jaccard Accuracy : ",dtree_jaccard)
dtree_f1 = f1_score(test_y, Pred_dtree, average='weighted')
print("F1 Score : ",dtree_f1)


# # SVM Evaluation Metrics- Test Data

# In[64]:


Pred_SVM = svm.predict(test_X)

SVM_jaccard = jaccard_score(test_y, Pred_SVM,pos_label = "PAIDOFF")
print("Jaccard Accuracy : ",SVM_jaccard)
SVM_f1 = f1_score(test_y, Pred_SVM, average="weighted")
print("F1 Score : ",SVM_f1)


# # Logistic Evaluation Metrics- Test Data

# In[65]:


Pred_LR = l_reg.predict(test_X)
yhat_prob_LR = l_reg.predict_proba(test_X)
yhat_prob_LR
LR_jaccard = jaccard_score(test_y, Pred_LR,pos_label = "PAIDOFF")
print("Jaccard Accuracy : ",LR_jaccard)
LR_f1 = f1_score(test_y, Pred_LR,average="weighted")
print("F1 Score : ",LR_f1)
LR_LogLoss = log_loss(test_y, yhat_prob_LR)
print("Log Loss : ",LR_LogLoss)


# In[72]:


Jaccard = [knn_jaccard, dtree_jaccard, SVM_jaccard, LR_jaccard]
f1score = [knn_f1, dtree_f1, SVM_f1, LR_f1]
Logloss = ['N/A', 'N/A', 'N/A', LR_LogLoss]

Report = pd.DataFrame(Jaccard, index=['KNN','Decision Tree','SVM','Logistic Regression'])
Report.columns = ['Jaccard']
Report.insert(loc=1, column='F1-score', value=f1score)
Report.insert(loc=2, column='LogLoss', value=Logloss)
Report.columns.name = 'Algorithm'
Report


# In[ ]:





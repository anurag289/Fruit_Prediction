#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[3]:


fruit=pd.read_csv("fruits.csv")


# In[4]:


fruit.head()


# In[5]:


fruit.describe()


# In[6]:


plt.scatter(x=fruit['wt'],y=fruit['sphericity'])
plt.show()


# In[7]:


#here I have speparated the featurs and target values for better prediction
X=fruit[['wt','sphericity']]
y=fruit['label']


# In[8]:


X.head()


# In[9]:


y.head()


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[11]:


plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Apple'],c='red',label='Train apple')
plt.legend()
plt.show()


# In[12]:


plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Orange'],c='orange',label='Train orange')
plt.legend()
plt.show()


# In[13]:


plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Apple'],c='red',label='Train apple')
plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Orange'],c='orange',label='Train orange')

plt.legend()
plt.show()

#we can celarly see from the plot that 


# In[14]:


plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Apple'],c='red',label='Train apple')
plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Orange'],c='orange',label='Train orange')
plt.scatter(x='sphericity',y='wt',data=X_test,c='green',label='test data',marker='*',s=100)


plt.legend()
plt.show()

#we can celarly see from the plot that 


# In[15]:


from sklearn.neighbors import KNeighborsClassifier  # importing KNN algorithm 
knn = KNeighborsClassifier(n_neighbors=5,weights='distance') # using the neighbour as 5 and weights as ditance
knn.fit(X_train,y_train)#traing with the data available
y_predict=knn.predict(X_test)
print(y_train.value_counts())
confusion_matrix(y_test,y_predict)


# In[16]:


def plot_fruit():
    plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Apple'],c='red',label='Train apple')
    plt.scatter(x='sphericity',y='wt',data=X_train[y_train=='Orange'],c='orange',label='Train orange')
    plt.scatter(x='sphericity',y='wt',data=X_test[y_test==y_predict],c='green',label='correct classification')
    plt.scatter(x='sphericity',y='wt',data=X_test[y_test!=y_predict],c='yellow',label='wrong classification')
    plt.xlabel("Sphericity")
    plt.ylabel("weight")
    plt.legend()
    plt.show()


# In[17]:


#here I have speparated the featurs and target values for better prediction
X=fruit[['wt','sphericity']]
y=fruit['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier  # importing KNN algorithm 
knn = KNeighborsClassifier(n_neighbors=5,weights='distance') # using the neighbour as 5 and weights as ditance
knn.fit(X_train,y_train)#traing with the data available
y_predict=knn.predict(X_test)
print(y_train.value_counts())
print(confusion_matrix(y_test,y_predict))
plot_fruit()


# In[18]:


#here I have speparated the featurs and target values for better prediction
X=fruit[['wt','sphericity']]
y=fruit['label']
scaling=MinMaxScaler()
X=pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier  # importing KNN algorithm 
knn = KNeighborsClassifier(n_neighbors=5,weights='distance') # using the neighbour as 5 and weights as ditance
knn.fit(X_train,y_train)#traing with the data available
y_predict=knn.predict(X_test)
print(y_train.value_counts())
print(confusion_matrix(y_test,y_predict))
plot_fruit()


# In[19]:


X = fruit[['wt','sphericity']]
y = fruit['label']
scaling = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
print(X.describe())
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
print(y_predict)
print(confusion_matrix(y_test,y_predict))
plot_fruit()


# ## KNN wihout scaling and weights = uniform

# In[20]:


X = fruit[['wt','sphericity']]
y = fruit['label']
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN without scaling")
plt.show()


# ## KNN with scaling weights = uniform

# In[21]:


X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN with scaling")
plt.show()


# ## KNN with scaling and Startify enabled and weights = uniform

# In[22]:


X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN with scaling, startify enabled")
plt.show()


# ## KNN with scaling and stratify enabled with weights = distance

# In[23]:


X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k,weights='distance')
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN with scaling,starify enabled and weights=distance")
plt.show()


# ## KNN with scaling and stratify enabled with weights = distance and applying euclidian and manhatten formula

# In[33]:


X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k,weights='distance')
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN with scaling, startify, weights =distance and formula used in euclidiean")
plt.show()

X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
accuracy=[]
r = range(1,20,2)
for k in r:
    clf = KNeighborsClassifier(n_neighbors=k,weights='distance',p=1)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    accuracy.append(clf.score(X_test,y_test))
    
plt.plot(r,accuracy,marker='o')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("KNN with scaling, startify, weights =distance and formula used in manahatten")
plt.show()


# ## by regressing the module multiple time found that its better to make our model use with below specs
# 
# ###Scaling enabled
# ###Stratify enabled
# ###weights = distance
# ###formula = euclidiean

# In[31]:


#we will now pickel the highest accuract model for further deployment

X = fruit[['wt','sphericity']]
y = fruit['label']
Scaler = MinMaxScaler()
X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
print(X_train.head())
Knear = KNeighborsClassifier(n_neighbors=5,weights='distance')
Knear.fit(X_train,y_train)
print(Knear.score(X_test,y_test))

'''plt.plot(r,accuracy,marker='o')

knn.fit(X_train,y_train)#traing with the data available
y_predict=knn.predict(X_test)
print(y_train.value_counts())
confusion_matrix(y_test,y_predict)'''

prediction=Knear.predict([[0.614458,0.443038]])
output=prediction[0]
print(output)
if output=="Orange":
   print("Thats an Orange")
else:
   print("Thats an apple which keeps dcotor away")



print(X_train.head())
print(y_train.head())
import pickle
#opening the file where I want to store the data; wb is write byte mode
file=open('Knearforfruits.pkl','wb')

#dummp the info in that file
pickle.dump(Knear, file)


# In[37]:


#we will now pickel the highest accuract model for further deployment without scaling

X = fruit[['wt','sphericity']]
y = fruit['label']
#Scaler = MinMaxScaler()
#X = pd.DataFrame(scaling.fit_transform(X),columns=['sphericity','wt'])
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,stratify=y)
print(y_train.value_counts())
print(X_train.head())
Knear = KNeighborsClassifier(n_neighbors=5,weights='distance')
Knear.fit(X_train,y_train)
print(Knear.score(X_test,y_test))

'''plt.plot(r,accuracy,marker='o')

knn.fit(X_train,y_train)#traing with the data available
y_predict=knn.predict(X_test)
print(y_train.value_counts())
confusion_matrix(y_test,y_predict)'''

prediction=Knear.predict([[162,0.839]])
output=prediction[0]
print(output)
if output=="Orange":
   print("Thats an Orange")
else:
   print("Thats an apple which keeps dcotor away")



print(X_train.head())
print(y_train.head())
import pickle
#opening the file where I want to store the data; wb is write byte mode
file=open('Knearforfruits.pkl','wb')

#dummp the info in that file
pickle.dump(Knear, file)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"C:\Users\K.NAGI REDDY\Downloads\project final\newdataset.csv")
data.head(10)


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Rumor_label', data=data)
plt.title('Distribution of Rumor Labels')
plt.xlabel('Rumor Label')
plt.ylabel('Count')
plt.show()


# In[7]:


text_source_counts = data['source'].value_counts()
plt.figure(figsize=(8, 6))
text_source_counts.plot(kind='pie',autopct='%1.1f%%')
plt.title('Count of Text Sources')
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Rumor_label' column to string data type
data['Rumor_label'] = data['Rumor_label'].astype(str)

plt.figure(figsize=(8, 6))
sns.countplot(x='Rumor_label', data=data)
plt.title('Distribution of Rumor Labels')
plt.xlabel('Rumor Label')
plt.ylabel('Count')
plt.show()


# In[9]:


X = data['text']
y = data['Rumor_label']


# In[10]:


print(X,"\n\n\n",y)


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Fill NaN values in the 'text' column with an empty string
data['text'].fillna('', inplace=True)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])


# In[ ]:





# In[12]:


X_vectorized


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


# In[14]:


from sklearn.metrics import accuracy_score,classification_report



from sklearn.svm import LinearSVC

S_classifier = LinearSVC()
S_classifier.fit(X_train, y_train)
y_pred = S_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)*100

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[15]:


from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)*100

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[16]:


from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)*100

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[17]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)*100

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[18]:


models = ['LinearSVC', 'Random Forest', 'KNN', 'Decision Tree']
accuracies = [svm_accuracy, rf_accuracy, knn_accuracy, dt_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Classification Models')
plt.ylim(0, 100)
plt.show()

best_model_index = accuracies.index(max(accuracies))
best_model = models[best_model_index]
print(f"The best model is '{best_model}' with an accuracy of {max(accuracies)}")


# In[19]:


classifier = LinearSVC()
classifier.fit(X_train, y_train)


# In[20]:


y_pred = classifier.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, xticklabels=['True', 'False'], yticklabels=['True', 'False'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:


example_text = [input("Enter rumor: ")]
example_text_vectorized = vectorizer.transform(example_text)
prediction = loaded_model.predict(example_text_vectorized)
print("Example prediction:", prediction)


# In[ ]:





# In[ ]:





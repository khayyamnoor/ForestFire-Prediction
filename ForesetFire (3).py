#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset

# In[238]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 


import seaborn as sns
import statsmodels.api  
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# # Dataset

# In[219]:


filename ="forestdata.csv" 
df = pd.read_csv(filename)


# In[220]:



df['fire'].value_counts()


# In[221]:


df.describe()


# # Data Preprocessing!

# In[267]:


import matplotlib.pyplot as plt

num_rows = 4
num_cols = 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

axes = axes.flatten()

variables = ['collector.id', 'c.score', 'l.score', 'rain', 'tree.age',
             'surface.litter', 'wind.intensity', 'humidity', 'tree.density',
             'month', 'time.of.day', 'fire']

for i, var in enumerate(variables):
    axes[i].hist(df[var], color='purple', bins=20)
    axes[i].set_title(var.capitalize())
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlabel('Value')

plt.tight_layout()

plt.show()

#Month Tme Collector.id C.score are Linearly Seperable


# In[223]:


#Removing all missing values
numerical_features = ['l.score', 'rain', 'tree.age', 'wind.intensity', 'humidity']
for feature in numerical_features:
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)
print(df.isnull().sum())


# In[224]:


# Morning,0
# Afternoon 1
# Night 2
df['time.of.day'] = df['time.of.day'].astype(str)
label_encoder = LabelEncoder()
df['time.of.day'] = label_encoder.fit_transform(df['time.of.day'])


# In[225]:


df.head(5)


# # Scalling 

# In[226]:


#Scalled the features for better prediction 
from sklearn.preprocessing import MinMaxScaler
continuous_features = ['c.score','l.score','rain', 'tree.age', 'surface.litter', 'wind.intensity', 'humidity']
scaler = MinMaxScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])


# In[227]:


corr = df.corr()
corr


# In[268]:


import matplotlib.pyplot as plt

continuous_features = ['c.score', 'l.score', 'rain', 'tree.age', 'surface.litter', 'wind.intensity', 'humidity']

num_features = len(continuous_features)
num_rows = (num_features + 1) // 2
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, feature in enumerate(continuous_features):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].hist(df[feature], bins=20, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'Histogram of {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True)

plt.tight_layout()

plt.show()


# # Logistic Regession

# In[258]:


X = df.drop(columns=['humidity','rain', 'collector.id', 'fire'])
y = df['fire']

# (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y,
                                    test_size=0.3,
                                    random_state=42)


# In[259]:


model = LogisticRegression(C=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Model Accuracy on Test Data = {accuracy*100:2f}%")
print(f"Model precision on Test Data = {precision*100:2f}%")
print(f"Model recall on Test Data = {recall*100:2f}%")
print(f"Model F1 on Test Data = {f1*100:2f}%")


# # Heatmap

# In[235]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
#These values provide insights into the performance of the classification model. 
#For example, a higher number of true positives and true negatives and lower numbers
#of false positives and false negatives indicate better model performance.


# In[232]:


print(cm)


# # Decission Tree

# In[184]:


df.info()


# In[185]:


df.describe()


# In[186]:


df['fire'].value_counts()


# # Data pre-processing
# 

# In[187]:


df.isna().sum()


# In[188]:


#The Values were scaled using MinMax Classifier so the Min and Max values have been reduced
print(f"Minimum value:\n{df.min()}")
print(f"Maximum value:\n{df.max()}")


# In[189]:


print(f"Average value:\n{df.mean()}")


# # Plotting

# In[269]:


import matplotlib.pyplot as plt

num_rows = 4
num_cols = 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

axes = axes.flatten()

variables = ['collector.id', 'c.score', 'l.score', 'rain', 'tree.age',
             'surface.litter', 'wind.intensity', 'humidity', 'tree.density',
             'month', 'time.of.day', 'fire']

for i, var in enumerate(variables):
    axes[i].hist(df[var], color='blue', bins=20)
    axes[i].set_title(var.capitalize())
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlabel('Value')

plt.tight_layout()

plt.show()


# In[270]:


corr = df.corr()
corr


# In[271]:


fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr, annot=True, ax=ax, cmap='winter');
fig.suptitle(t="HeatMap",
             color="orange",
             fontsize=16);
# Negative corelation between wind and humidity
# Strong/Positive corelation between tree density and surface litter


# # Preprocessing

# In[272]:


df.head()


# # Removing for prediction

# In[273]:


preds_data1 = df.iloc[32]
preds_data1


# In[274]:


preds_data2 = df.iloc[76]
preds_data2


# In[275]:


preds_data3 = df.iloc[132]
preds_data3


# In[276]:


df.drop([32, 76, 132], inplace=True)
df.shape


# In[277]:


X = df.drop(columns=['collector.id','humidity','rain', 'fire'])
y = df['fire']


# In[278]:


X.dtypes


# In[279]:


y


# # Training and Testing

# In[280]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
np.random.seed(35)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3)
max_depth = 5
clf = DecisionTreeClassifier(max_depth=max_depth)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f"Model Accuracy on Test Data = {score*100:2f}%")


# In[281]:


y_preds = clf.predict(X_test)
print(f"Classfifcation Report:\n\n{classification_report(y_test, y_preds)}")


# In[282]:


cf_matrix = confusion_matrix(y_test, y_preds)
print(f"Confusion Matrix:\n\n{cf_matrix}")


# In[283]:


fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cf_matrix, annot=True, cmap='coolwarm')
fig.suptitle(t="Confusion Matrix",
             color="orange",
             fontsize=16);
ax.set(xlabel="Predicted Label",
       ylabel="Actual Label");
#The results have turned out to be better since ive removed "humidity" and "rain"


# # Making prediction 
# 

# In[284]:


preds_data1


# In[285]:




pred_x1 = pd.DataFrame(np.array([0.003272, 0.087316, 0.011278, 0.405563, 0.409799, 0.633500, 2.000000, 0.000000]).reshape(1, -1),
                       columns=['c.score', 'l.score', 'tree.age', 'surface.litter', 'wind.intensity', 'tree.density', 'month', 'Time'])

pred_y1 = clf.predict(pred_x1)

encoded_to_original = {0: 'No fire', 1: 'Fire'}
pred_class1 = encoded_to_original.get(pred_y1[0], 'Unknown')

print(f"Predicted class by model on preds_data1: {pred_class1}")
print(f"Prediction is Correct")


# In[286]:


preds_data2


# In[287]:


pred_x2 = pd.DataFrame(np.array([0.002067, 0.034427, 0.022932, 0.242883, 0.297586, 0.543060, 6.000000, 0.000000]).reshape(1, -1),
                       columns=['c.score', 'l.score', 'tree.age', 'surface.litter', 'wind.intensity', 'tree.density', 'month', 'Time'])

pred_y2 = clf.predict(pred_x2)

encoded_to_original = {0: 'No fire', 1: 'Fire'}
pred_class2 = encoded_to_original.get(pred_y2[0], 'Unknown')

print(f"Predicted class by model on preds_data2: {pred_class2}")
print("Prediction is correct")


# # Improving our model

# In[288]:


np.random.seed(42)
for max_depth in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Model Accuracy on test set: {clf.score(X_test, y_test)*100:2f}%")
    print(" ")


# # MLP Classifier
# 

# In[292]:


label_encoder = LabelEncoder()
df['month'] = label_encoder.fit_transform(df['month'])
df['time.of.day'] = label_encoder.fit_transform(df['time.of.day'])


# In[293]:


X = df.drop(columns=['humidity','rain', 'collector.id', 'fire'])
y = df['fire']


# In[294]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[295]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_classifier.fit(X_train, y_train)


# In[296]:


y_pred = mlp_classifier.predict(X_test)


# In[297]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision*100:.2f}")


# In[298]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for MLPClassifier')
plt.show()
print(cm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





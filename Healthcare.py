import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df =  pd.read_csv(r'C:\Users\HP\Desktop\Machine learning\ml_practice\parkinson_disease.csv')
# print(df)

# Now, time to data cleaning
# print(df.info())
# So this dataset have this type of information

# Now, convert the float values into int

df1 = df.astype('int64')
# print(df1.info())

# Check the null values

# print(df1.isnull().sum())
# there is not any null values in this dataset

# Check the duplicate values
# df2 = df1[df1.duplicated()]
# print(df2)
# print(df1.duplicated().sum())

# Divide the dependent and independent features
x = df1.iloc[:, :-1]
y = df1.iloc[:, -1]
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42)
# print(x_train)
# print(x_test)
# print(y_train)

# Display correlation heatmap
plt.figure(figsize =(12,10))
correlation_matrix = x_train.corr()
strong_corr = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
sns.heatmap(correlation_matrix, annot = False, cmap = plt.cm.CMRmap_r)
# plt.show()

# Now create the function of coorelation
def coorelation(dataset, threshold):
    col_corr = set()
    correlation_matrix = dataset.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i,j]) > threshold:
                colname = correlation_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = coorelation(df1, 0.7)
# print(len(set(corr_features)))

x_train_corrfeatures = x_train.drop(corr_features, axis = 1)
x_test_corrfeatures = x_test.loc[:,x_train_corrfeatures.columns]
# df_filtered = df.loc[scaler_fit.index]
# x_test = df1.drop(corr_features, axis = 1)
# print(x_train_corrfeatures.shape)
# print(x_test.shape)

# just want to make simpler data values in dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
scaler = MinMaxScaler()
scaler_fit = scaler.fit_transform(x_train_corrfeatures,x_test_corrfeatures)
# If I want to select the best features from the dataset upto 40 features
best_features = SelectKBest(chi2, k = 30)
# df_filtered = df1.loc[scaler_fit.index]
df_filtered = df1.loc[:scaler_fit.shape[0]-1]
best_features.fit(scaler_fit,df_filtered['class'])
features_selectors = best_features.get_support()
filtered_data = x_train_corrfeatures.loc[:,features_selectors]
filtered_data['class'] = df1['class']
df2 = filtered_data
print(df2.shape)

#  to check the data is imbalance or not
target_occurence = df2['class'].value_counts()
plt.pie(target_occurence, labels = target_occurence.index,autopct = '%1.1f%%')
# plt.show()
#The output is 74.6 and 25.4 which is not a balanced dataset


# Now, I will balance the dataset
from imblearn.under_sampling import RandomUnderSampler
# Taking some parameters
sampler = RandomUnderSampler(sampling_strategy = 0.5,random_state = 42)
x_resample, y_resample = sampler.fit_resample(x_train_corrfeatures, y_train)
print(x_resample.shape)
print(y_resample.shape)              
# x_test_filtered = x_test.loc[:,x_train.columns]
# print(x_test_filtered.shape)
# print(x_resample.shape)
# print(x_train)
# print(x_test)
# Now, I will apply the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Now, I will check the all three models, these are my base models
models = [LogisticRegression(),RandomForestClassifier(n_estimators=100,max_depth=2),XGBClassifier(n_estimators=100,learning_rate=0.5)]
for i in range(len(models)):
    models[i].fit(x_resample,y_resample)
    # what's the model been choosen
    print(f'{models[i]}, ')
    # predict the data
    y_pred = models[i].predict(x_test_corrfeatures)
    # print(y_pred)
    c = accuracy_score(y_test,y_pred)
    print("Accuracy score of listed models: ", c)
    # i want to check overfitting
    x_train_predict = models[i].predict(x_train_corrfeatures)
    training_accuracy = accuracy_score(y_train,x_train_predict)
    print("training accuracy of listed models: ", training_accuracy)


# We will choose the XGBoost classifier model 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix  
parameters = {
    'C': [0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear']
             }
grid_search = GridSearchCV(LogisticRegression(),parameters,cv = 5,scoring='accuracy')
grid_search.fit(x_train_corrfeatures,y_train)
print(grid_search.best_params_)
predicted_labels = grid_search.predict(x_test_corrfeatures)
# check the accuracy
predicted_accuracy = accuracy_score(y_test,predicted_labels)
print("The predicted accuracy of logistic model:", predicted_accuracy)
x_trained_predict = grid_search.predict(x_train_corrfeatures)
trained_accuracy = accuracy_score(y_train,x_trained_predict)
print("The training accuracy of logistic model: ",trained_accuracy)

# Now, I will check the classification report
print("The classification report of logistic model: ", classification_report(y_test,predicted_labels))
# Now, I will check the confusion matrix
cm = confusion_matrix(y_test,predicted_labels)

# plot the sns heatmap for confusion matrix
plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True, cmap = 'Blues',xticklabels=['yes','no'],yticklabels=['yes','no'])
plt.xlabel('y_test')
plt.ylabel('predicted_labels')
plt.title('Confusion matrix heatmap')
plt.show()
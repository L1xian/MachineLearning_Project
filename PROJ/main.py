# Imports
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib

from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import ListedColormap

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,f1_score, hinge_loss, precision_score, recall_score )
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

## 1: Data Analysis and Visualization
data = pd.read_csv('natural_disasters_dataset.csv')
print(data.info())
print(data.head(10))

# grouping the data by 'Continent' and 'Disaster Type' and counting the frequency of the Disaster Type
disaster_counts = data.groupby(['Continent', 'Disaster Type']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
disaster_counts.plot(kind='bar', stacked=True, colormap='inferno')
plt.xlabel('Continent')
plt.ylabel('Frequency')
plt.title('Frequency of Disaster Types by Continent')
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Displaying the top 5 continents with the highest frequency of disasters
top_continents = disaster_counts.sum(axis=1).nlargest(5)
print("Continents where the Disasters occurred more, in descending order:")
print(top_continents)

# Displaying the frequency count
plt.figure(figsize=(14, 8))
disaster_type_counts = data['Disaster Type'].value_counts().sort_values(ascending=False)
sns.barplot(x=disaster_type_counts, y=disaster_type_counts.index)
for index, value in enumerate(disaster_type_counts):
    plt.text(value, index, str(value), va='center', fontsize=10, color='black', ha='left')
plt.title('Distribution of Disaster Types')
plt.xlabel('Count')
plt.ylabel('Disaster Type')
plt.show()

# Visualizing correlation HeatMap for Numerical features
plt.figure(figsize=(16, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#top 5 disaster types over the years
plt.figure(figsize=(14, 8))
# Filter the data for the top 5 disaster types
top_disaster_types = data['Disaster Type'].value_counts().nlargest(5).index
filtered_data = data[data['Disaster Type'].isin(top_disaster_types)]
# Group the filtered data by 'Start Year' and 'Disaster Type' and calculate the count
disaster_type_counts = filtered_data.groupby(['Start Year', 'Disaster Type']).size().unstack(fill_value=0)
# Plot time series for each top 5 disaster type
for disaster_type in disaster_type_counts.columns:
    plt.plot(disaster_type_counts.index, disaster_type_counts[disaster_type], label=disaster_type)
plt.title('Time Series Analysis of Top 5 Disaster Types Over the Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend()
plt.show()


### 2: Data Preprocessing

# 1/ Checking for Missing values
print("Null values:")
print(data.isnull().sum())
# 2/ Replacing Missing values
data.replace('nan', np.nan, inplace=True)
numerical_cols = data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
categorical_cols = data.select_dtypes(include='object').columns
imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
# checking data integrity
print(data.isnull().sum())
# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
# Identify and remove single-valued columns
non_single_valued_columns = data.columns[data.nunique() > 1]
filtered_data = data[non_single_valued_columns]

### 3: Feature Selection

# using mutual information
X = data.drop('Disaster Type', axis=1)
y = data['Disaster Type']
X['Year'] = X['Year'].astype(float) # including 'Year' feature
mutual_info = mutual_info_classif(X, y)
feature_importance = pd.Series(mutual_info, index=X.columns)
# Selecting top 10 features
selected_features = feature_importance.nlargest(10).index
if 'Year' not in selected_features:
    selected_features = selected_features.append(pd.Index(['Year']))
# Printing the selected features
X_selected = X[selected_features]
print(X_selected.columns)

# Loading the selected features into a new CSV file
selected_features = [
    'Year',  'Dis Mag Scale','Dis Mag Value', 'Country', 'Longitude', 'Latitude', 'Disaster Type'
]
data_selected = data[selected_features]
data_selected.to_csv('preprocessed_data.csv', index=False)
data_selected = pd.read_csv('preprocessed_data.csv')
data_selected.head(10)
print(data_selected.isnull().sum())

### 4: Model Development

# 1/Random Forest Classifier
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=50,  # Number of trees in the forest
    max_depth=None,    # Maximum depth of the tree
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# 2/K-NN
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(
    n_neighbors=5,  # Number of neighbors to use
    weights='uniform',  # Weight function used in prediction
    algorithm='auto',  # Algorithm used to compute the nearest neighbors
)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# 3/Navie Bayes
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluating the performance of Random Forest
print("Random Forest Classifier Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall (Sensitivity):", recall_score(y_test, y_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("\n")

# Evaluating the performance of K-NN
print("K-Nearest Neighbor (K-NN)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_knn, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_knn, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted'))
print("\n")

# Evaluating the performance of Navie Bayes
print("Navie Bayes-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_nb, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("\n")

### Step 6: Tuning

# Checking the class distribution
print(data_selected['Disaster Type'].value_counts())
# Separating feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']
# Initializing the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
# Fitting and applying the oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)
# Checking the new class distribution
print(pd.Series(y_resampled).value_counts())
# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# Standardizing/Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Retrain

# Training the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
# Training the K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
# Training the Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

# Evaluating the performance of Random Forest
print("Random Forest Classifier Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_rf, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("\n")
# Evaluating the performance of K-NN
print("K-Nearest Neighbor (K-NN)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_knn, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_knn, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted'))
print("\n")
# Evaluating the performance of Navie Bayes
print("Navie Bayes-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_nb, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("\n")

### voting Ensemble combining 3 models
# Combining individual models to form a Ensemble Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
nb_model = GaussianNB()
# Creating a hard voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('knn', knn_model),
    ('nb', nb_model)
], voting='hard')
# Fitting the ensemble model
ensemble_model.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
# Evaluate the ensemble model
print("\nEnsemble Model (Hard Voting):")
print("F1 Score:", f1_score(y_test, y_pred_ensemble, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_ensemble, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_ensemble, average='weighted'))

# Creating a soft voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('knn', knn_model),
    ('nb', nb_model)
], voting='soft')
# Fitting the ensemble model
ensemble_model.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
# Evaluate the ensemble model
print("\nEnsemble Model (Hard Voting):")
print("F1 Score:", f1_score(y_test, y_pred_ensemble, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_ensemble, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_ensemble, average='weighted'))


# Models and their respective perfromance metrics
models = ['Random Forest', 'K-NN', 'Naive Bayes', 'Ensemble Hard-Voting', 'Ensemble Soft-Voting']
f1_scores = [0.9519, 0.9258, 0.6297, 0.91947, 0.93508 ]
accuracies = [0.9538, 0.9313, 0.6532, 0.92361, 0.93905 ]
recalls = [0.9538, 0.9313, 0.6532, 0.92361, 0.93905]
precisions = [0.9552, 0.9316, 0.7167, 0.92917, 0.94320]

bar_width = 0.2
index = np.arange(len(models))

fig, ax = plt.subplots(figsize=(14, 6))
bar1 = ax.bar(index - bar_width, f1_scores, bar_width, label='F1 Score')
bar2 = ax.bar(index, accuracies, bar_width, label='Accuracy')
bar3 = ax.bar(index + bar_width, recalls, bar_width, label='Recall')
bar4 = ax.bar(index + 2 * bar_width, precisions, bar_width, label='Precision')


ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation Metrics')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(models)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

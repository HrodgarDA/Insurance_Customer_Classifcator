# # ML model for cross selling customer classification

# %% RESOURCES LOADING
import sys
import subprocess
import pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from collections import Counter

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# %% DATA LOADING
data_path =  '/Users/rugg/Documents/GitHub/Insurance_Customer_Classifcator/Insurance_data.csv'
dataframe = pd.read_csv(data_path)
print(dataframe.head())

RS = np.random.randint(0, 100)

# %% EXPLORATIVE DATA ANALYSIS
print(f'DATAFRAME INFO:')
print(dataframe.info())
print("_"*70)
print(f'DATAFRAME NULL COUNT: \n {dataframe.isnull().sum()}')
print("_"*70)
print(f'DATAFRAME DESCRIPTION: \n {dataframe.describe()}')
print("_"*70)

# Target Variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Response', data=dataframe, hue='Response')
plt.title('Target class value distribution')
plt.show()

# Distribution of feature values
features = dataframe.columns.drop(['Response', 'id'])
n_features = len(features)
n_cols = 5
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
fig.suptitle('Distribution of feature Values', fontsize=25)

for i, feature in enumerate(features):
    ax = axes[i // n_cols, i % n_cols]
    if dataframe[feature].dtype in ['int64', 'float64']:
        sns.histplot(data=dataframe, x=feature, hue='Response', kde=True, ax=ax, legend=False)
    else:
        sns.countplot(data=dataframe, x=feature, hue='Response', ax=ax, legend=False)
    ax.set_title(feature)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% OUTLIER DETECTION AND CLEANING
#Detection
def plot_outliers(df, threshold=3):
    fig, axs = plt.subplots(len(df.columns), figsize=(12, 4*len(df.columns)), constrained_layout=True)
    
    for i, feature in enumerate(df.columns):
        if np.issubdtype(df[feature].dtype, np.number):
            unique_values = df[feature].unique()
            if len(unique_values) > 2 or not (0 in unique_values and 1 in unique_values):
                z_scores = np.abs(stats.zscore(df[feature]))
                outliers = df.loc[np.abs(z_scores) > threshold]
                
                axs[i].boxplot(df[feature], vert=False, showfliers=False)
                axs[i].scatter(outliers[feature], [1] * len(outliers), color='red', label='Outliers')
                axs[i].set_xlabel(feature)
            else:
                axs[i].axis('off')
        else:
            axs[i].axis('off')
    
    plt.show()

plot_outliers(dataframe)
print('\nOutliers detected in Annual_Premium feature')

# Cleaning
def remove_outliers(df, threshold=3):
    df_clean = df.copy()
    for feature in df.columns:
        if np.issubdtype(df[feature].dtype, np.number):
            unique_values = df[feature].unique()
            if len(unique_values) > 2 or not (0 in unique_values and 1 in unique_values):
                z_scores = np.abs(stats.zscore(df[feature]))
                df_clean = df_clean[np.abs(z_scores) <= threshold]
    return df_clean

cleaned_df = remove_outliers(dataframe)

print("Shape of previous dataframe:", dataframe.shape)
print("Shape of cleaned dataframe:", cleaned_df.shape)
percentage = round((len(cleaned_df)/len(dataframe))*100, 2)
print(f"Percentage of values kept: {percentage}%")

print('Number of null values in target variable:', cleaned_df['Response'].isna().sum())
print(cleaned_df.head(1))

# %% FEATURE ENGINEERING

# Label Encoding
cleaned_df['Vehicle_Age'] = cleaned_df['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})
cleaned_df['Gender_Flag'] = cleaned_df['Gender'].map({'Male': 1, 'Female': 0})
cleaned_df['Vehicle_Damage'] = cleaned_df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
cleaned_df['Previously_Insured'] = cleaned_df['Previously_Insured'].astype(int)

cleaned_df.drop(['Gender', 'id'], axis=1, inplace=True)
cleaned_df = cleaned_df[[col for col in cleaned_df if col != 'Response'] + ['Response']]

print(cleaned_df.head(3))

# Correlation matrix
correlation_matrix = cleaned_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# %% TARGET CLASS BALANCING
X = cleaned_df.drop('Response', axis=1)
y = cleaned_df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)

print("Original classes distribution:")
print(Counter(y_train))
print(f'Minor class percentage before balancing = {round(sum(y_train==1)/len(y_train)*100,2)}%')

# Balancing pipeline
over_under_pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.6, random_state=42)),
    ('under', RandomUnderSampler(sampling_strategy=0.75, random_state=42))
])

X_train_balanced, y_train_balanced = over_under_pipeline.fit_resample(X_train, y_train)

print("\nClass distribution after over-under sampling:")
print(Counter(y_train_balanced))
print(f'Minor class percentage after balancing = {round(sum(y_train_balanced==1)/len(y_train_balanced)*100,2)}%')

# %% FEATURE SELECTION

def compare_feature_selection_methods(X, y, n_features=5, cv=5):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    selectors = [
        ('SelectKBest', SelectKBest(score_func=f_classif, k=n_features)),
        ('RFE', RFE(LogisticRegression(max_iter=1000), n_features_to_select=n_features, step=1)),
        ('Lasso', SelectFromModel(Lasso(alpha=0.1), max_features=n_features))
    ]
    
    results = {}

    for name, selector in selectors:
        try:
            selector.fit(X_scaled, y)
            mask = selector.get_support()
            best_features = X.columns[mask].tolist()
            X_selected = X_scaled[best_features]

            scores = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=cv)

            results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores) * 2,
                'best_features': best_features
            }

            print(f"{name}: mean CV score = {results[name]['mean_score']:.4f} (+/- {results[name]['std_score']:.4f})")
            print(f"Best features: {', '.join(best_features)}\n")

        except Exception as e:
            print(f"An error occurred with {name}: {str(e)}")

    all_features = set()
    for result in results.values():
        all_features.update(result['best_features'])

    print(f"Union of all selected features: {', '.join(sorted(all_features))}")

    return results

results = compare_feature_selection_methods(X_train_balanced, y_train_balanced, n_features=5, cv=5)

# DATAFRAME SCALING
selected_features = list(set(results['SelectKBest']['best_features'] +
                             results['RFE']['best_features'] +
                             results['Lasso']['best_features']))

X_selected = X_train_balanced[selected_features]
X_test_selected = X_test[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(X_test_selected)

# %% MODEL SELECTION AND EVALUATION
# Custom function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print('\nScores Report:')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return pd.DataFrame({'Model': [model.__class__.__name__], 'Accuracy': [accuracy], 
                         'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]})

# MODEL EVALUATION
# Logistic Regression
log_reg = LogisticRegression(random_state=RS)
log_reg_performances = evaluate_model(log_reg, X_train_scaled, y_train_balanced, X_test_scaled, y_test)
# %%
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=RS)
rf_model_performances = evaluate_model(rf_model, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %%
# SVM
svm_model = SVC(kernel='rbf', random_state=RS)
svm_model_performances = evaluate_model(svm_model, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %%
# Naive Bayes
nb_model = GaussianNB()
nb_model_performances = evaluate_model(nb_model, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %%
# Performance comparison
performances_df = pd.concat([log_reg_performances, rf_model_performances, 
                             svm_model_performances, nb_model_performances], ignore_index=True)
print(performances_df)

# %% [markdown]
# ## Conclusioni e Raccomandazioni

# Sulla base dei risultati ottenuti, possiamo trarre le seguenti conclusioni:

# 1. Il modello Random Forest ha mostrato le migliori prestazioni complessive, con il più alto F1-score.
# 2. La Regressione Logistica e SVM hanno mostrato prestazioni simili, leggermente inferiori al Random Forest.
# 3. Il modello Naive Bayes ha avuto le prestazioni più basse tra i modelli testati.

# Raccomandazioni:
# 1. Utilizzare il modello Random Forest per le previsioni finali.
# 2. Considerare l'ottimizzazione degli iperparametri per il Random Forest per migliorare ulteriormente le prestazioni.
# 3. Esplorare tecniche di feature engineering più avanzate per creare nuove caratteristiche informative.
# 4. Valutare l'impatto delle singole caratteristiche sul modello finale per comprendere meglio i fattori chiave che influenzano la risposta del cliente.

# Prossimi passi:
# 1. Implementare il modello Random Forest ottimizzato in produzione.
# 2. Monitorare le prestazioni del modello nel tempo e aggiornarlo regolarmente con nuovi dati.
# 3. Utilizzare le previsioni del modello per personalizzare le strategie di marketing e le offerte di cross-selling.
# %% ML model for cross selling customer classification

# %% RESOURCES LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# %% DATA LOADING
data_path = '/Users/rugg/Documents/GitHub/Insurance_Customer_Classifcator/Insurance_data.csv'
dataframe = pd.read_csv(data_path)
print(dataframe.head())

RS = 42  # Fixed random state for reproducibility

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

over_under_pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.6, random_state=RS)),
    ('under', RandomUnderSampler(sampling_strategy=0.75, random_state=RS))
])

X_train_balanced, y_train_balanced = over_under_pipeline.fit_resample(X_train, y_train)

print("\nClass distribution after over-under sampling:")
print(Counter(y_train_balanced))
print(f'Minor class percentage after balancing = {round(sum(y_train_balanced==1)/len(y_train_balanced)*100,2)}%')

# %% FEATURE SELECTION AND SCALING
def compare_feature_selection_methods(X, y, n_features=5, cv=5):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    selectors = [
        ('SelectKBest', SelectKBest(score_func=f_classif, k=n_features)),
        ('RFE', RFE(LogisticRegression(max_iter=1000, random_state=RS), n_features_to_select=n_features, step=1)),
        ('Lasso', SelectFromModel(Lasso(alpha=0.1, random_state=RS), max_features=n_features))
    ]
    
    results = {}

    for name, selector in selectors:
        try:
            selector.fit(X_scaled, y)
            mask = selector.get_support()
            best_features = X.columns[mask].tolist()
            X_selected = X_scaled[best_features]

            scores = cross_val_score(LogisticRegression(max_iter=1000, random_state=RS), X_selected, y, cv=cv)

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

selected_features = list(set().union(*[results[method]['best_features'] for method in results]))

X_train_selected = X_train_balanced[selected_features]
X_test_selected = X_test[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# %% MODEL SELECTION AND EVALUATION
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

# %% Logistic Regression
log_reg = LogisticRegression(random_state=RS)
log_reg_performances = evaluate_model(log_reg, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %% Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=RS)
rf_model_performances = evaluate_model(rf_model, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %% Naive Bayes
nb_model = GaussianNB()
nb_model_performances = evaluate_model(nb_model, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

# %% Performance comparison
performances_df = pd.concat([log_reg_performances, rf_model_performances, nb_model_performances], ignore_index=True)
print(performances_df)

# %% RANDOM FOREST FINE TUNING
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.5),
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=RS)

halving_search = HalvingRandomSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_candidates=50,
    factor=3,
    cv=5,
    random_state=RS,
    n_jobs=-1,
    scoring='f1'
)

halving_search.fit(X_train_scaled, y_train_balanced)

print("Best hyperparameters found:")
print(halving_search.best_params_)

best_rf = halving_search.best_estimator_
rf_optimized_performances = evaluate_model(best_rf, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

performances_df = pd.concat([performances_df, rf_optimized_performances], ignore_index=True)
performances_df.iloc[-1, performances_df.columns.get_loc('Model')] = 'Random Forest (Optimized)'
print(performances_df)

feature_importance = best_rf.feature_importances_
feature_names = selected_features

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, orient='h')
plt.title('Feature Importance in Optimized Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# %% Conclusions

"""
Based on the results obtained, we can draw the following conclusions:

1. The Random Forest model showed the best overall performance, with the highest F1-score.
2. The Naive Bayes model had the lowest performance among the tested models.
3. Without fine-tuning, the maximum accuracy obtained was 0.785, with a high number of false positives and negatives.
4. The optimized Random Forest model improved upon the base model, demonstrating the effectiveness of hyperparameter tuning.

Next steps:
1. Implement the optimized Random Forest model in production.
2. Monitor the model's performance over time and update it regularly with new data.
3. Use the model's predictions to personalize marketing strategies and cross-selling offers.
4. Consider exploring more advanced ensemble methods or deep learning approaches for potential further improvements.
5. Conduct a thorough analysis of misclassified instances to gain insights into areas where the model can be improved.
"""
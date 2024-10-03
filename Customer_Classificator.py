# %% [markdown]
# # Previsione di opportunità di Cross Sell di assicurazioni
# 
# Questo notebook analizza un dataset di assicurazioni per prevedere se i clienti sarebbero interessati ad acquistare un'assicurazione per il proprio veicolo.

# %% [markdown]
# ## Importazione delle librerie

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# %% [markdown]
# ## Caricamento e esplorazione dei dati

# %%
# Caricamento dei dati
df = pd.read_csv('insurance.csv')

# Visualizzazione delle prime righe del dataset
print(df.head())

# Informazioni sul dataset
print(df.info())

# Statistiche descrittive
print(df.describe())

# Controllo dei valori mancanti
print(df.isnull().sum())

# %% [markdown]
# ## Visualizzazione dei dati

# %%
# Distribuzione della variabile target
plt.figure(figsize=(8, 6))
sns.countplot(x='Response', data=df)
plt.title('Distribuzione della variabile target')
plt.show()

# Correlazione tra le variabili numeriche
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice di correlazione')
plt.show()

# Distribuzione dell'età
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribuzione dell\'età')
plt.show()

# Relazione tra età e premio annuale
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Annual_Premium', hue='Response', data=df)
plt.title('Relazione tra età e premio annuale')
plt.show()

# Distribuzione del premio annuale per risposta
plt.figure(figsize=(10, 6))
sns.boxplot(x='Response', y='Annual_Premium', data=df)
plt.title('Distribuzione del premio annuale per risposta')
plt.show()

# %% [markdown]
# ## Preparazione dei dati

# %%
# Separazione delle caratteristiche e della variabile target
X = df.drop(['id', 'Response'], axis=1)
y = df['Response']

# Codifica one-hot per le variabili categoriche
X = pd.get_dummies(X, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])

# Divisione in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scalatura delle caratteristiche numeriche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Selezione delle caratteristiche

# %%
# Selezione delle caratteristiche più importanti
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Ottieni i nomi delle caratteristiche selezionate
selected_features = X.columns[selector.get_support()].tolist()
print("Caratteristiche selezionate:", selected_features)

# %% [markdown]
# ## Gestione dello sbilanciamento delle classi

# %%
# Gestione dello sbilanciamento delle classi con SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# %% [markdown]
# ## Definizione e valutazione dei modelli

# %%
# Definizione dei modelli da valutare
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Funzione per valutare un modello
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

# Valutazione dei modelli
results = {}
for name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train_resampled, y_train_resampled, X_test_selected, y_test)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

# Visualizzazione dei risultati
results_df = pd.DataFrame(results).T
print(results_df)

# %% [markdown]
# ## Visualizzazione dei risultati

# %%
# Visualizzazione grafica dei risultati
plt.figure(figsize=(12, 6))
results_df.plot(kind='bar')
plt.title('Confronto delle performance dei modelli')
plt.xlabel('Modelli')
plt.ylabel('Punteggio')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Selezione e valutazione finale del miglior modello

# %%
# Selezione del miglior modello
best_model_name = results_df['F1-score'].idxmax()
best_model = models[best_model_name]

# Addestramento del miglior modello sui dati completi
best_model.fit(X_train_resampled, y_train_resampled)

# Valutazione finale del miglior modello
y_pred = best_model.predict(X_test_selected)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice di confusione - {best_model_name}')
plt.xlabel('Previsto')
plt.ylabel('Reale')
plt.show()

print(f"Il miglior modello è: {best_model_name}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from joblib import Parallel, delayed
import time
import logging
import cProfile
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
logging.basicConfig(filename='program_log.txt', level=logging.INFO)
# Load dataset
def log_time_and_step(step_name):
    logging.info(f"{step_name}: {time.ctime()}")

def load_data():
    log_time_and_step("Loading Data")
    df = pd.read_csv('OASIS_CLEANED_frequencyEncode.csv')
    return df


def prepare_data(df):
    log_time_and_step("Preparing Data")
    #df = df.sample(frac=0.5, random_state=1)
    X = df.drop(['BENE_ID','ASMT_ID', 'READMISSION'], axis=1).values
    y = df['READMISSION'].values
    log_time_and_step("Balancing Data")
    class_0, class_1 = df['READMISSION'].value_counts()

    # Balancing dataset using undersampling
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X, y)

    log_time_and_step("Balanced")
    df_balanced = pd.DataFrame(X_balanced, columns=df.drop(['BENE_ID','ASMT_ID', 'READMISSION'], axis=1).columns)
    df_balanced['READMISSION'] = y_balanced
    #df_sampled = df_balanced.sample(frac=0.5, random_state=1)
    X = df_balanced.drop('READMISSION', axis=1).values
    y = df_balanced['READMISSION'].values
    log_time_and_step("Splitting Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_time_and_step("Scaling Data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_time_and_step("Imputing Nan Data")

    imputer = SimpleImputer(strategy='most_frequent')  # use 'median', 'most_frequent', or 'mean'
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed, y_train, y_test

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def train_model(model, param_grid, X_train, X_test, y_train, y_test, model_name):
    """Train a model with optional hyperparameter tuning"""
    log_time_and_step(f"Starting hyperparameter tuning for {model_name}")
    best_model, best_params = tune_hyperparameters(model, param_grid, X_train, y_train)
    log_time_and_step(f"Starting training for {model_name}")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    log_time_and_step(f"Finished training for {model_name}")
    return model_name, y_pred, best_params

def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
        'Random Forest': (RandomForestClassifier(n_jobs=-1, random_state=123),
                          {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=123),
                              {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]})
    }

    results = Parallel(n_jobs=-1)(
        delayed(train_model)(model, param_grid, X_train, X_test, y_train, y_test, name)
        for name, (model, param_grid) in models.items()
    )

    return results



def save_results(results, X_train, X_test, y_train, y_test):
    log_time_and_step("Saving Results")
    results_dict = {}
    best_params_dict = {}  # Ensure to initialize this dictionary

    for model_name, y_pred in results:
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        if model_name == 'Ridge Regression':
            metrics = {
                'Training Size': len(X_train),
                'Test Size': len(X_test),
                'Class 0 Count': sum(y_test == 0),
                'Class 1 Count': sum(y_test == 1),
                'MSE': mean_squared_error(y_test, y_pred),
                'R^2': r2_score(y_test, y_pred),
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            }
        else:
            metrics = {
                'Training Size': len(X_train),
                'Test Size': len(X_test),
                'Class 0 Count': sum(y_test == 0),
                'Class 1 Count': sum(y_test == 1),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            }

        results_dict[model_name] = metrics
        best_params_dict[model_name] = best_params  # Assuming `best_params` is defined elsewhere

    results_df = pd.DataFrame(results_dict).T
    results_df.to_csv('modeling_bl_results.csv', index=True)

    best_params_df = pd.DataFrame.from_dict(best_params_dict, orient='index')
    best_params_df.to_csv('best_hyperparameters.csv', index=True)


def main():
    start_time = time.time()
    log_time_and_step("Start Program")

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    results = evaluate_models(X_train, X_test, y_train, y_test)
    save_results(results, X_train, X_test, y_train, y_test)

    log_time_and_step("End Program")
    logging.info(f"Total Time: {time.time() - start_time:.2f} seconds")

# Profile the code
cProfile.run('main()')

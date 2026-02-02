import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import logging
import time
import os

# Configure logging (Optional, kept if needed for debug)
logging.basicConfig(filename='program_log_fs_levels.txt', level=logging.INFO)

def log_time_and_step(step_name):
    logging.info(f"{step_name}: {time.ctime()}")

# --- Feature Selection Methods ---

def variance_threshold(X_train, y_train, X_test, y_test):
    log_time_and_step("Variance Threshold")
    vt = VarianceThreshold(threshold=0.1)
    X_vt = vt.fit_transform(X_train)
    vt_features = vt.get_support(indices=True)
    vt_scores = vt.variances_[vt_features]
    return {
        'method': 'Variance Threshold',
        'indices': vt_features,
        'scores': vt_scores,
        'score_type': 'Variance'
    }

def lasso_selection(X_train, y_train, X_test, y_test):
    log_time_and_step("Lasso Selection")
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    lasso_features = np.where(lasso.coef_ != 0)[0]
    lasso_scores = lasso.coef_[lasso_features]
    return {
        'method': 'Lasso',
        'indices': lasso_features,
        'scores': lasso_scores,
        'score_type': 'Coefficient'
    }

def rf_importance(X_train, y_train, X_test, y_test):
    log_time_and_step("Random Forest Importance")
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    rf_features = np.argsort(rf_importances)[::-1]
    rf_scores = rf_importances[rf_features]
    return {
        'method': 'Random Forest Importance',
        'indices': rf_features,
        'scores': rf_scores,
        'score_type': 'Importance'
    }

def permutation_importance_rf(X_train, y_train, X_test, y_test):
    log_time_and_step("Permutation Importance (RF)")
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    perm_importance_rf = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=1)
    perm_features_rf = perm_importance_rf.importances_mean.argsort()[::-1]
    perm_scores_rf = perm_importance_rf.importances_mean[perm_features_rf]
    return {
        'method': 'Permutation Importance (RF)',
        'indices': perm_features_rf,
        'scores': perm_scores_rf,
        'score_type': 'Importance'
    }

def permutation_importance_ridge(X_train, y_train, X_test, y_test):
    log_time_and_step("Permutation Importance (Ridge)")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    perm_importance_ridge = permutation_importance(ridge, X_test, y_test, n_repeats=10, random_state=1)
    perm_features_ridge = perm_importance_ridge.importances_mean.argsort()[::-1]
    perm_scores_ridge = perm_importance_ridge.importances_mean[perm_features_ridge]
    return {
        'method': 'Permutation Importance (Ridge)',
        'indices': perm_features_ridge,
        'scores': perm_scores_ridge,
        'score_type': 'Importance'
    }

def rfe_rf(X_train, y_train, X_test, y_test):
    log_time_and_step("RFE (RF)")
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rfe_rf = RFE(estimator=rf, n_features_to_select=5, step=1)
    rfe_rf.fit(X_train, y_train)
    rfe_features_rf = rfe_rf.get_support(indices=True)
    rf.fit(X_train[:, rfe_features_rf], y_train)
    rfe_scores_rf = rf.feature_importances_
    return {
        'method': 'RFE (RF)',
        'indices': rfe_features_rf,
        'scores': rfe_scores_rf,
        'score_type': 'Importance'
    }

def rfe_ridge(X_train, y_train, X_test, y_test):
    log_time_and_step("RFE (Ridge)")
    ridge = Ridge(alpha=1.0)
    rfe_ridge = RFE(estimator=ridge, n_features_to_select=5, step=1)
    rfe_ridge.fit(X_train, y_train)
    rfe_features_ridge = rfe_ridge.get_support(indices=True)
    rfe_scores_ridge = rfe_ridge.coef_[rfe_features_ridge] if hasattr(ridge, 'coef_') else np.zeros(len(rfe_features_ridge))
    return {
        'method': 'RFE (Ridge)',
        'indices': rfe_features_ridge,
        'scores': rfe_scores_ridge,
        'score_type': 'Coefficient'
    }

def sfs_knn(X_train, y_train, X_test, y_test):
    log_time_and_step("SFS (KNN)")
    knn = KNeighborsRegressor(n_neighbors=5)
    sfs_knn = SequentialFeatureSelector(knn, n_features_to_select=5, direction='forward')
    sfs_knn.fit(X_train, y_train)
    sfs_features_knn = sfs_knn.get_support(indices=True)
    return {
        'method': 'SFS (KNN)',
        'indices': sfs_features_knn,
        'scores': [None] * len(sfs_features_knn),
        'score_type': 'N/A'
    }

def sfs_ridge(X_train, y_train, X_test, y_test):
    log_time_and_step("SFS (Ridge)")
    ridge = Ridge(alpha=1.0)
    sfs_ridge = SequentialFeatureSelector(ridge, n_features_to_select=5, direction='forward')
    sfs_ridge.fit(X_train, y_train)
    sfs_features_ridge = sfs_ridge.get_support(indices=True)
    return {
        'method': 'SFS (Ridge)',
        'indices': sfs_features_ridge,
        'scores': [None] * len(sfs_features_ridge),
        'score_type': 'N/A'
    }

def gb_importance(X_train, y_train, X_test, y_test):
    log_time_and_step("Gradient Boosting Importance")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=1)
    gb.fit(X_train, y_train)
    gb_importances = gb.feature_importances_
    gb_features = np.argsort(gb_importances)[::-1]
    gb_scores = gb_importances[gb_features]
    return {
        'method': 'GB Importance',
        'indices': gb_features,
        'scores': gb_scores,
        'score_type': 'Importance'
    }

def lgb_importance(X_train, y_train, X_test, y_test):
    log_time_and_step("LightGBM Importance")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=1)
    lgb_model.fit(X_train, y_train)
    lgb_importances = lgb_model.feature_importances_
    lgb_features = np.argsort(lgb_importances)[::-1]
    lgb_scores = lgb_importances[lgb_features]
    return {
        'method': 'LightGBM Importance',
        'indices': lgb_features,
        'scores': lgb_scores,
        'score_type': 'Importance'
    }

def xgb_importance(X_train, y_train, X_test, y_test):
    log_time_and_step("XGBoost Importance")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=1)
    xgb_model.fit(X_train, y_train)
    xgb_importances = xgb_model.feature_importances_
    xgb_features = np.argsort(xgb_importances)[::-1]
    xgb_scores = xgb_importances[xgb_features]
    return {
        'method': 'XGBoost Importance',
        'indices': xgb_features,
        'scores': xgb_scores,
        'score_type': 'Importance'
    }

def catboost_importance(X_train, y_train, X_test, y_test):
    log_time_and_step("CatBoost Importance")
    catboost_model = cb.CatBoostRegressor(iterations=100, random_state=1, verbose=0)
    catboost_model.fit(X_train, y_train)
    catboost_importances = catboost_model.feature_importances_
    catboost_features = np.argsort(catboost_importances)[::-1]
    catboost_scores = catboost_importances[catboost_features]
    return {
        'method': 'CatBoost Importance',
        'indices': catboost_features,
        'scores': catboost_scores,
        'score_type': 'Importance'
    }

# --- Mapping Dictionaries ---

AVAILABLE_METHODS = {
    'Variance Threshold': variance_threshold,
    'Lasso': lasso_selection,
    'Random Forest Importance': rf_importance,
    'Permutation Importance (RF)': permutation_importance_rf,
    'Permutation Importance (Ridge)': permutation_importance_ridge,
    'RFE (RF)': rfe_rf,
    'RFE (Ridge)': rfe_ridge,
    'SFS (KNN)': sfs_knn,
    'SFS (Ridge)': sfs_ridge,
    'Gradient Boosting Importance': gb_importance,
    'LightGBM Importance': lgb_importance,
    'XGBoost Importance': xgb_importance,
    'CatBoost Importance': catboost_importance
}

METHOD_CATEGORIES = {
    "Filter Methods": [
        "Variance Threshold"
    ],
    "Wrapper Methods": [
        "RFE (RF)", "RFE (Ridge)", "SFS (KNN)", "SFS (Ridge)"
    ],
    "Embedded Methods": [
        "Lasso", "Random Forest Importance", "Gradient Boosting Importance", 
        "LightGBM Importance", "XGBoost Importance", "CatBoost Importance"
    ],
    "Permutation Methods": [
        "Permutation Importance (RF)", "Permutation Importance (Ridge)"
    ]
}

# --- Helper Functions ---

def save_feature_selection_results(feature_selection_results, feature_names, output_dir='.', dataset_name='Custom', level='NA', dataset_level='NA'):
    """
    Saves feature selection results to a CSV file.
    """
    # Fix for when passed a single result dictionary (from app) vs list of results (from main)
    if isinstance(feature_selection_results, dict):
        feature_selection_results = [feature_selection_results]
        
    log_time_and_step(f"Saving Feature Selection Results for {dataset_name}")
    rows = []

    for result in feature_selection_results:
        method = result['method']
        indices = result['indices']
        scores = result['scores']
        score_type = result['score_type']

        for idx, score in zip(indices, scores):
            rows.append({
                'Level': level,
                'Dataset': dataset_name,
                'Method': method,
                'Feature Index': idx,
                'Feature Name': feature_names[idx],
                'Score': score,
                'Score Type': score_type
            })

    fs_results_df = pd.DataFrame(rows)
    
    # Create directory for this dataset level if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file path in the dataset level directory
    results_file = os.path.join(output_dir, 'Feature_Selection_Results.csv')
    
    try:
        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
            combined_df = pd.concat([existing_df, fs_results_df], ignore_index=True)
            combined_df.to_csv(results_file, index=False)
        else:
            fs_results_df.to_csv(results_file, index=False)
    except Exception as e:
        # Fallback to write if duplicate handling fails
        fs_results_df.to_csv(results_file, index=False)

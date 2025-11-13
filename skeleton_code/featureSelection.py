import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import logging
import time
from joblib import Parallel, delayed


logging.basicConfig(filename='program_log_fs.txt', level=logging.INFO)

def log_time_and_step(step_name):
    logging.info(f"{step_name}: {time.ctime()}")

def load_data():
    log_time_and_step("Loading Data")
    df = pd.read_csv('OASIS_CLEANED_frequencyEncode.csv')
    return df

def prepare_data(df):
    log_time_and_step("Preparing Data")
    df = df.sample(frac=0.5, random_state=1)
    X = df.drop(['BENE_ID','ASMT_ID', 'READMISSION'], axis=1).values
    y = df['READMISSION'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def perform_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    log_time_and_step("Performing Feature Selection")

    def variance_threshold():
        log_time_and_step("Variance Threshold")
        print("Running Variance Threshold")
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

    def lasso_selection():
        log_time_and_step("Lasso Selection")
        print("Running Lasso Selection")
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

    def rf_importance():
        log_time_and_step("Random Forest Importance")
        print("Running Random Forest Importance")
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

    def permutation_importance_rf():
        log_time_and_step("Permutation Importance (RF)")
        print("Running Permutation Importance (RF)")
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

    def permutation_importance_ridge():
        log_time_and_step("Permutation Importance (Ridge)")
        print("Running Permutation Importance (Ridge)")
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

    def rfe_rf():
        log_time_and_step("RFE (RF)")
        print("Running RFE (RF)")
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        rfe_rf = RFE(estimator=rf, n_features_to_select=5, step=1)
        rfe_rf.fit(X_train, y_train)
        rfe_features_rf = rfe_rf.get_support(indices=True)
        rf.fit(X_train[:, rfe_features_rf], y_train)  # Refit on selected features
        rfe_scores_rf = rf.feature_importances_
        return {
            'method': 'RFE (RF)',
            'indices': rfe_features_rf,
            'scores': rfe_scores_rf,
            'score_type': 'Importance'
        }

    def rfe_ridge():
        log_time_and_step("RFE (Ridge)")
        print("Running RFE (Ridge)")
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

    def sfs_knn():
        log_time_and_step("SFS (KNN)")
        print("Running SFS (KNN)")
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

    def sfs_ridge():
        log_time_and_step("SFS (Ridge)")
        print("Running SFS (Ridge)")
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

    def gb_importance():
        log_time_and_step("Gradient Boosting Importance")
        print("Running Gradient Boosting Importance")
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

    def lgb_importance():
        log_time_and_step("LightGBM Importance")
        print("Running LightGBM Importance")
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

    def xgb_importance():
        log_time_and_step("XGBoost Importance")
        print("Running XGBoost Importance")
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

    def catboost_importance():
        log_time_and_step("CatBoost Importance")
        print("Running CatBoost Importance")
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

    # Parallelizing feature selection methods
    methods = [
        variance_threshold,
        lasso_selection,
        rf_importance,
        permutation_importance_rf,
        permutation_importance_ridge,
        #rfe_rf,
        rfe_ridge,
        sfs_knn,
        sfs_ridge,
        gb_importance,
        lgb_importance,
        xgb_importance,
        catboost_importance
    ]

    results = Parallel(n_jobs=-1)(delayed(method)() for method in methods)

    return results

def save_feature_selection_results(feature_selection_results, feature_names):
    log_time_and_step("Saving Feature Selection Results")
    rows = []

    for result in feature_selection_results:
        method = result['method']
        indices = result['indices']
        scores = result['scores']
        score_type = result['score_type']

        for idx, score in zip(indices, scores):
            rows.append({
                'Method': method,
                'Feature Index': idx,
                'Feature Name': feature_names[idx],
                'Score': score,
                'Score Type': score_type
            })

    fs_results_df = pd.DataFrame(rows)
    fs_results_df.to_csv('Feature_Selection_Results.csv', index=False)

def main():
    start_time = time.time()
    log_time_and_step("Start Program")

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    feature_names = df.drop(['BENE_ID','ASMT_ID', 'READMISSION'], axis=1).columns
    feature_selection_results = perform_feature_selection(X_train, y_train, X_test, y_test, feature_names)
    save_feature_selection_results(feature_selection_results, feature_names)
    compare_regression_coefficients(X_train, y_train)

    log_time_and_step("End Program")
    logging.info(f"Total Time: {time.time() - start_time:.2f} seconds")

# Profile the code
cProfile.run('main()')

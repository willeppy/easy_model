import numpy as np
import pandas as pd
import json
import math

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

import shap


# gridsearchCV throws lots of warnings so supress bc not critical
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def create_model(df_x, 
                df_y, 
                task = "classification", 
                classifier_name = None, 
                score_metrics = None, 
                calc_feat_importance = False, 
                normalize = True):
    '''
    Main method for model creation 

    Params
    --------
        df_x: pd.DataFrame
            Features 
        
        df_y: pd.DataFrame
            Target
        
        task: str 
            "classification" or "regression"

        classifier_name: str
            'RF', 'NN', 'DT', 'LR', 'SVM' or None (tries all)
    
    Returns
    --------
        Response: dict
            model, metrics, feat importances 
    '''
    
    if normalize:
        df_x = normalize_data(df_x)

    # get default metrics
    if score_metrics is None:
        if task == "classification":
            score_metrics = ["accuracy"]
        else:
            score_metrics = ["neg_root_mean_squared_error"]
    
    # run classifier 
    if classifier_name is not None:
        best_clf, metrics_dict = train_model(df_x, df_y, task, classifier_name, score_metrics)
    else:
         best_clf, metrics_dict = train_all_models(df_x, df_y, task, score_metrics)

    # get feat importance
    if calc_feat_importance:
        feat_importance = get_feature_importance(classifier_name, best_clf, df_x)
    else:
        feat_importance = None

    response = {"model": best_clf,  
                "metrics": metrics_dict,
                "feat_importance": feat_importance
                }

    return response


def normalize_data(df_x):
    df_x = (df_x - df_x.mean()) / df_x.std() 
    return df_x

def train_all_models(df_x, df_y, task, score_metrics):
    ms = ['RF', 'NN', 'DT', 'LR', 'SVM']

    print(f"Trying models: {ms}")

    eval_crit = score_metrics[0]

    best_model = None
    best_metric = None
    for m in ms:
    
        clf, metrics = train_model(df_x, df_y, task, m, score_metrics)
        print(f"\t{m}: {metrics}")

        if best_metric is None or metrics[eval_crit] > best_metric[eval_crit]:
            best_model = clf
            best_metric = metrics
    
    return best_model, best_metric

def train_model(X, y, task, classifier_name, scoring):
        
    defaults = {
        "NN": {"hidden_layer_sizes": [(100), (50, 25), (100, 50)], "learning_rate_init": [.01, .001], "max_iter": [100]},
        "DT": {"max_depth": [None, 10]},
        "SVM": {"kernel": ["linear", "poly", "rbf",]}
    }

    params = {}

    if task == "classification":
        if classifier_name == 'RF':
            clf = RandomForestClassifier()
            params = {"n_estimators": [10, 100], "criterion": ["gini", "entropy"]}
        
        if classifier_name == 'NN':
            clf = MLPClassifier()
            params = defaults["NN"]
        
        if classifier_name == 'DT':
            clf = DecisionTreeClassifier()
            params = defaults["DT"]
        
        if classifier_name == 'LR':
            clf = LogisticRegression()
            params = {"penalty": ["l1", "l2", "elasticnet"], "solver": ["lbfgs", "saga"], "l1_ratio":[.5], "max_iter":[200]}
        
        if classifier_name == "SVM":
            clf = SVC()
            params = defaults["SVM"]
    
    else: # regression
        if classifier_name == 'RF':
            clf = RandomForestRegressor()
            params = {"n_estimators": [10, 100], "criterion": ["mse", "mae"]}

        if classifier_name == 'NN':
            clf = MLPRegressor()
            params = defaults["NN"]

        if classifier_name == 'DT':
            clf = DecisionTreeRegressor()
            params = defaults["DT"]

        if classifier_name == 'LR':
            clf = ElasticNetCV()
            params = {"l1_ratio": [.1, .5, .7, .9, .95, .99, 1]}

        if classifier_name == "SVM":
            clf = SVR()
            params = defaults["SVM"]
    
    # evaluates with both scoring metrics but selects the best_model with the first
    cv = GridSearchCV(clf, params, scoring=scoring, refit=scoring[0], cv=5)
    cv.fit(X, y)
    
    # best estimator
    best_clf = cv.best_estimator_
    
    # get average metrics
    metric_df = pd.DataFrame(cv.cv_results_)
    metrics = {}
    
    for m in scoring:
        _val = metric_df[metric_df[f"rank_test_{scoring[0]}"] == 1][f"mean_test_{m}"].values[0]
        metrics[m] = _val
        
    return best_clf, metrics


'''
If using LR, then take the model coefficients.
Otherwise, compute SHAP values.

SHAP: https://shap.readthedocs.io/en/latest/api.html

'''
def get_feature_importance(classifier_name, clf, X):
   
    if classifier_name == 'LR':
        ### taks abs val of coefficients of linear model
        feat_imps = abs(clf.coef_.flatten()).tolist() 
    
    elif classifier_name in ["RF", "DT"]:
        ### TreeExplainer is super fast for tree methods
        
        background = shap.maskers.Independent(X, max_samples=500)
        explainer = shap.TreeExplainer(clf, masker=background)
        shap_vals = explainer(X)
        
        feat_imps = shap_vals.abs.values.mean(axis=0)[:,0].tolist()
    
    else: # ["NN", "SVM"]
        
        ### generic method -- super slow!!
        # background = shap.maskers.Independent(X, max_samples=100)
        # explainer = shap.Explainer(clf, background)
        # shap_vals = explainer(X)
        # feat_imps = shap_vals.abs.values.mean(axis=0).tolist()
        
        ### Kernel Explainer is a bit faster
        med = np.median(X, axis=0).reshape(1, -1)
        explainer = shap.KernelExplainer(clf.predict, med)
        shap_vals = explainer.shap_values(X)
        feat_imps = abs(shap_vals.mean(axis=0)).tolist()
    
    # d = dict(zip(X.columns.tolist(), feat_imps))
    d = []

    for f, imp in zip(X.columns.tolist(), feat_imps):
        d.append({"feat": f, "imp": imp})
    
    return d
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import json

def train_model(X, y, parameters: dict=None):
    default =dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
        eval_metric='logloss',
    )

    if parameters:
        default.update(parameters)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(**default)
    model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=True)
    
    return model, X_test, Y_test

def evaluate_model(model, X_test, Y_test) -> dict:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(Y_test, y_pred)
    class_report = classification_report(Y_test, y_pred)
    conf_matrix = confusion_matrix(Y_test, y_pred)
    logloss = log_loss(Y_test, y_pred_proba)
    brier_score = brier_score_loss(Y_test, y_pred_proba)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
        'log_loss': logloss,
        'brier_score_loss': brier_score,
        'roc_auc_score': roc_auc
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

#save_model implemented manually in notebook thanks to redundancy with xgboost's built in method.

def load_model(path: str):
    model = XGBClassifier()
    model.load_model(path)
    print(f"Model loaded from {path}")
    return model

def predict(model, X) -> pd.Series:
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name='xFG')
from model_handler import ModelFactory
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import pandas as pd

# ⭐ Custom AUPRC metric
def lgbm_auprc_metric(y_true, y_pred):
    """AUPRC metric for LightGBM (higher is better)"""
    score = average_precision_score(y_true, y_pred)
    return 'auprc', score, True  # (name, value, is_higher_better)

class LGBMFactory(ModelFactory):
    """Factory for LightGBM using sklearn API with AUPRC metric"""
    
    @property
    def model_name(self):
        return "LightGBM"
    
    def get_params(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
    
    def create_model(self, params):
        return lgb.LGBMClassifier(**params)
    
    def fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Fit with AUPRC metric for early stopping"""
        
        # Ensure X_train has feature names
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        
        if X_val is not None and y_val is not None:
            # Ensure X_val has the same feature names
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val, columns=X_train.columns)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgbm_auprc_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ]
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def predict(self, model, X):
        if not isinstance(X, pd.DataFrame):
            # Get feature names from the model
            feature_names = model.feature_name_
            X = pd.DataFrame(X, columns=feature_names)
        
        return model.predict_proba(X)[:, 1]
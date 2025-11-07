from model_handler import ModelFactory
import xgboost as xgb

class XGBFactory(ModelFactory):
    """Factory for creating and configuring XGBoost models"""
    
    @property
    def model_name(self):
        return "XGBoost"
    
    def get_params(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'tree_method': 'hist',
            'device': 'cuda:0',
            'eval_metric': 'aucpr',
            'early_stopping_rounds': 50
        }
    
    def create_model(self, params):
        """Create XGBoost model with GPU support"""
        return xgb.XGBClassifier(**params)
    
    def fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Fit the XGBoost model with optional early stopping"""
        if X_val is not None and y_val is not None:
            # Fit with validation set for early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            # Fit without early stopping
            model.fit(X_train, y_train, verbose=False)
            print(f"  Trained {model.n_estimators} trees (no validation set)")
        
        return model
    
    def predict(self, model, X):
        """Make predictions using XGBoost with GPU"""
        dmatrix = xgb.DMatrix(X)
        return model.get_booster().predict(dmatrix)
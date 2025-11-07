from abc import ABC, abstractmethod
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
import optuna


class DataPreprocessor:
    """Handles all data preprocessing: numpy conversion, scaling, and sampling"""
    
    def __init__(self, X_train_raw, y_train, X_test_raw, y_test, sampler=None, X_val_raw=None, y_val=None):
        self.X_train_raw = self._to_numpy(X_train_raw)
        self.y_train = self._to_numpy(y_train)
        self.X_test_raw = self._to_numpy(X_test_raw)
        self.y_test = self._to_numpy(y_test)

        self.X_val_raw = self._to_numpy(X_val_raw) if X_val_raw is not None else None
        self.y_val = self._to_numpy(y_val) if y_val is not None else None
        self.use_validation = X_val_raw is not None and y_val is not None

        self.sampler = sampler
        self.scaler = StandardScaler()
        
        self.scaler.fit(self.X_train_raw)
        
        # Scale evaluation sets
        if self.use_validation:
            self.X_val_scaled = self.scaler.transform(self.X_val_raw)
        self.X_test_scaled = self.scaler.transform(self.X_test_raw)
    
    @staticmethod
    def _to_numpy(data):
        """Convert pandas to numpy for GPU compatibility"""
        if hasattr(data, 'values'):
            return data.values
        return np.asarray(data)
    
    def process_fold(self, train_idx, val_idx):
        """
        Process a single CV fold: Sample -> Scale -> Return numpy arrays.
        """
        # Split data
        X_train = self.X_train_raw[train_idx]
        X_val = self.X_train_raw[val_idx]
        y_train = self.y_train[train_idx]
        y_val = self.y_train[val_idx]
        
        # Fit scaler on training fold
        fold_scaler = StandardScaler()
        fold_scaler.fit(X_train)
        
        # Apply sampling if enabled
        if self.sampler:
            X_train, y_train = self.sampler.fit_resample(X_train, y_train)
        
        # Scale data
        X_train_scaled = fold_scaler.transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def process_full_training_set(self):
        """Apply sampling to full training set and scale"""
        if self.sampler:
            X_train_sampled, y_train_sampled = self.sampler.fit_resample(
                self.X_train_raw, self.y_train
            )
        else:
            X_train_sampled = self.X_train_raw
            y_train_sampled = self.y_train
        
        X_train_scaled = self.scaler.transform(X_train_sampled)
        return X_train_scaled, y_train_sampled
    
    def calculate_scale_pos_weight(self, y=None):
        if y is None:
            y = self.y_train
        return (len(y) - y.sum()) / y.sum()
    
    
    def get_validation_data(self):
        """Get processed validation set"""
        if not self.use_validation:
            raise ValueError("No validation set available")
        
        # Apply same preprocessing as training
        X_val_scaled = self.scaler.transform(self.X_val_raw)
        return X_val_scaled, self.y_val

class HyperparameterOptimiser:
    """Handles Optuna hyperparameter optimisation"""
    
    def __init__(self, model_factory, preprocessor, cache_manager, 
                n_trials, n_folds, random_state, data_tag, model_name):
        self.model_factory = model_factory
        self.preprocessor = preprocessor
        self.cache_manager = cache_manager
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.data_tag = data_tag
        self.model_name = model_name
    
    def optimise(self):
        """Run Optuna optimisation or load cached study"""
        study_key = f'{self.data_tag}_optimised_{self.model_name.lower()}'
        study = self.cache_manager.load_study(study_key)
        
        if study is None:
            print(f"\nOptimising {self.model_name} "
                  f"({self.n_trials} trials × {self.n_folds}-fold CV)...")
            
            study = optuna.create_study(
                direction='maximize',
                study_name=study_key,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            study.optimise(self._objective, n_trials=self.n_trials, 
                          show_progress_bar=True)
            self.cache_manager.save_study(study, study_key)
        else:
            print(f"\nLoaded cached study for {self.model_name}")
        
        return study
    
    def _objective(self, trial):
        """Optuna objective function for a single trial"""
        params = self.model_factory.get_params(trial)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                            random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in skf.split(
            self.preprocessor.X_train_raw, 
            self.preprocessor.y_train
        ):
            # Process fold
            X_tr_scaled, y_tr, X_val_scaled, y_val = self.preprocessor.process_fold(
                train_idx, val_idx
            )
            
            try:
                params_with_weight = params.copy()
                params_with_weight['scale_pos_weight'] = self.preprocessor.calculate_scale_pos_weight(y_tr)
                
                model = self.model_factory.create_model(params_with_weight)
                self.model_factory.fit_model(model, X_tr_scaled, y_tr, X_val_scaled, y_val)
                
                pred = self.model_factory.predict(model, X_val_scaled)
                score = average_precision_score(y_val, pred)
                scores.append(score)
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        return np.mean(scores)


class ModelTrainer:
    """Handles final model training with best parameters"""
    
    def __init__(self, model_factory, preprocessor, cache_manager, 
                data_tag, model_name, random_state):
        self.model_factory = model_factory
        self.preprocessor = preprocessor
        self.cache_manager = cache_manager
        self.data_tag = data_tag
        self.model_name = model_name
        self.random_state = random_state
    
    def train_final_model(self, best_params):
        """Train final model with best parameters on full training set"""
        # Prepare parameters
        params = best_params.copy()
        params['random_state'] = self.random_state
        
        # Process full training set
        X_train_scaled, y_train_sampled = self.preprocessor.process_full_training_set()
        
        # Update scale_pos_weight
        scale_pos_weight = self.preprocessor.calculate_scale_pos_weight(y_train_sampled)
        params['scale_pos_weight'] = scale_pos_weight
        
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Train or load model
        model_key = f'{self.data_tag}_optimised_{self.model_name.lower()}'
        model = self.cache_manager.load_model(model_key)
        
        if model is None:
            print("Training final model...")
            model = self.model_factory.create_model(params)
            
            if self.preprocessor.use_validation:
                X_val_scaled, y_val = self.preprocessor.get_validation_data()
                self.model_factory.fit_model(
                    model, X_train_scaled, y_train_sampled, 
                    X_val=X_val_scaled, y_val=y_val
                )
            else:
                # No validation set so train without early stopping
                self.model_factory.fit_model(model, X_train_scaled, y_train_sampled)
            
            self.cache_manager.save_model(model, model_key)
        else:
            print("Loaded cached model")
        
        return model
    
    def evaluate_model(self, model):
        """Evaluate model on validation set (if available) and test set"""
        results = {}
        
        # Evaluate on validation set if available
        if self.preprocessor.use_validation:
            preds_val = self.model_factory.predict(model, self.preprocessor.X_val_scaled)
            auprc_val = average_precision_score(self.preprocessor.y_val, preds_val)
            
            y_pred_binary = (preds_val >= 0.5).astype(int)
            precision_val = precision_score(self.preprocessor.y_val, y_pred_binary)
            recall_val = recall_score(self.preprocessor.y_val, y_pred_binary)
            
            print(f"\n{self.model_name} Validation AUPRC: {auprc_val:.4f}")
            print(f"{self.model_name} Validation Precision: {precision_val:.4f}")
            print(f"{self.model_name} Validation Recall: {recall_val:.4f}")
            
            results['validation_auprc'] = auprc_val
            results['validation_precision'] = precision_val
            results['validation_recall'] = recall_val
        
        # ALWAYS evaluate on test set
        preds_test = self.model_factory.predict(model, self.preprocessor.X_test_scaled)
        auprc_test = average_precision_score(self.preprocessor.y_test, preds_test)
        
        y_pred_binary = (preds_test >= 0.5).astype(int)
        precision_test = precision_score(self.preprocessor.y_test, y_pred_binary)
        recall_test = recall_score(self.preprocessor.y_test, y_pred_binary)
        
        print(f"\n{self.model_name} Test AUPRC: {auprc_test:.4f}")
        print(f"{self.model_name} Test Precision: {precision_test:.4f}")
        print(f"{self.model_name} Test Recall: {recall_test:.4f}\n")
        
        results['test_auprc'] = auprc_test
        results['test_precision'] = precision_test
        results['test_recall'] = recall_test
        
        return results

class ModelFactory(ABC):
    """Base class for creating and fitting models"""
    
    @property
    @abstractmethod
    def model_name(self):
        pass
    
    @abstractmethod
    def get_params(self, trial):
        pass
    
    @abstractmethod
    def create_model(self, params):
        pass
    
    @abstractmethod
    def fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Fit model with optional validation set for early stopping"""
        pass
    
    @abstractmethod
    def predict(self, model, X):
        """Make predictions with the model"""
        pass

class BoostModelManager:
    """Orchestrates the training pipeline"""
    
    def __init__(self, X_train_raw, y_train, X_test_raw, y_test, 
                 cache_manager, sampler, data_tag,
                 n_trials, n_folds, random_state, model_factory,
                 X_val_raw=None, y_val=None):
        
        self.model_factory = model_factory
        self.cache_manager = cache_manager
        self.data_tag = data_tag
        self.random_state = random_state
        
        # Initialise components
        self.preprocessor = DataPreprocessor(
            X_train_raw, y_train, X_test_raw, y_test, sampler,
            X_val_raw=X_val_raw, y_val=y_val
        )
        
        self.optimiser = HyperparameterOptimiser(
            model_factory, self.preprocessor, cache_manager,
            n_trials, n_folds, random_state, data_tag, 
            model_factory.model_name
        )
        
        self.trainer = ModelTrainer(
            model_factory, self.preprocessor, cache_manager,
            data_tag, model_factory.model_name, random_state
        )
        
        # Log configuration
        scale_pos_weight = self.preprocessor.calculate_scale_pos_weight()
        sampler_name = sampler.get_name() if sampler else "None"
        eval_set = "validation+test" if self.preprocessor.use_validation else "test only"
        print(f"Configuration: Sampling={sampler_name}, "
              f"scale_pos_weight={scale_pos_weight:.2f}, "
              f"eval_on={eval_set}\n")
        
        self.results = {}
    
    def train(self):
        """Execute complete training pipeline"""
        study = self.optimiser.optimise()
        
        print(f"\nBest CV AUPRC: {study.best_value:.4f}")
        print("\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key:20s}: {value}")
        
        # Train final model
        model = self.trainer.train_final_model(study.best_params)
        metrics = self.trainer.evaluate_model(model)
        
        # Store all results
        for metric_name, metric_value in metrics.items():
            result_key = f'{self.model_factory.model_name} {self.data_tag} {metric_name}'
            self.results[result_key] = metric_value
    
    def get_results(self):
        """Get evaluation results"""
        if not self.results:
            raise RuntimeError("Must call train() before get_results()")
        return self.results
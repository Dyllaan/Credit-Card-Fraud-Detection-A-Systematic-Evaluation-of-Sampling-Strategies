from typing import List, Optional, Type
import model_handler
from xgb_factory import XGBFactory
from lgbm_factory import LGBMFactory

def run_model(model_type, sampler, X_train_raw, y_train, X_test_raw, y_test, 
              cache_manager, data_tag, n_trials, n_folds, random_state,
              X_val_raw=None, y_val=None):
    
    if model_type == 'xgb':
        factory = XGBFactory()
    elif model_type == 'lgbm':
        factory = LGBMFactory()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    manager = model_handler.BoostModelManager(
        X_train_raw, y_train,
        X_test_raw, y_test,
        cache_manager,
        sampler=sampler,
        data_tag=data_tag,
        n_trials=n_trials,
        n_folds=n_folds,
        random_state=random_state,
        model_factory=factory,
        X_val_raw=X_val_raw,
        y_val=y_val
    )
    manager.train()
    return manager.get_results()


class SamplingOrchestrator:
    """Universal orchestrator that works with any sampler class"""
    
    MODEL_TYPES = ['xgb', 'lgbm']
    
    def __init__(self, sampler_class: Type, tag_prefix: str, requires_ratio: bool = True):
        """
        Args:
            sampler_class: The sampler class to instantiate
            tag_prefix: Prefix for data tags (e.g: 'smote', 'adasyn', 'no_sampling')
            requires_ratio: Whether this sampler needs a ratio parameter
        """
        self.sampler_class = sampler_class
        self.tag_prefix = tag_prefix
        self.requires_ratio = requires_ratio
        self.results = {}
    
    def run_sampling(self, X_train_raw, y_train, X_test_raw, y_test, 
                     cache_manager, n_trials, n_folds, random_state, 
                     ratios: Optional[List[float]] = None, X_val_raw=None, y_val=None, **sampler_kwargs):
        """
        Generic sampling execution that works for all samplers
        
        Args:
            ratios: List of ratios to test (None for no-sampling case)
            **sampler_kwargs: Additional arguments for sampler creation (like k_neighbors)
        """
        if not self.requires_ratio:
            ratios = [None]
        elif ratios is None:
            raise ValueError(f"{self.sampler_class.__name__} requires ratios to be specified")
        
        for ratio in ratios:
            # Build sampler kwargs
            init_kwargs = {'random_state': random_state, **sampler_kwargs}
            if ratio is not None:
                init_kwargs['ratio'] = ratio
            
            sampler = self.sampler_class(**init_kwargs)
            
            data_tag = self.tag_prefix if ratio is None else f'{self.tag_prefix}_{ratio}'
            
            # Run both model types
            for model_type in self.MODEL_TYPES:
                result_key = f'{data_tag}_{model_type}'
                self.results[result_key] = run_model(
                    model_type, sampler, X_train_raw, y_train, 
                    X_test_raw, y_test, cache_manager, data_tag, 
                    n_trials, n_folds, random_state,
                    X_val_raw=X_val_raw, y_val=y_val
                )
    
    def get_results(self):
        return self.results
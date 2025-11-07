from abc import ABC, abstractmethod
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours

class Sampler(ABC):
    """Abstract base class for sampling techniques"""
    
    @abstractmethod
    def fit_resample(self, X, y):
        """Apply sampling technique"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return sampler name for logging"""
        pass


class SMOTESampler(Sampler):
    """SMOTE sampling"""
    
    def __init__(self, ratio, random_state, k_neighbors=3):
        self.ratio = ratio
        self.random_state = random_state
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X, y):
        
        # Store column names if X is a DataFrame
        columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        fraud_count = y.sum()
        non_fraud_count = len(y) - fraud_count
        target_fraud_count = int(non_fraud_count * self.ratio)
        
        smote = SMOTE(
            sampling_strategy={1: target_fraud_count},
            random_state=self.random_state,
            k_neighbors=self.k_neighbors
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to DataFrame if original was DataFrame
        if columns is not None:
            X_resampled = pd.DataFrame(X_resampled, columns=columns)
            
        
        return X_resampled, y_resampled
    
    def get_name(self):
        return f"SMOTE_{self.ratio}:1"


class ADASYNSampler(Sampler):
    """ADASYN sampling"""
    
    def __init__(self, ratio, random_state):
        self.ratio = ratio
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        
        # Store column names if X is a DataFrame
        columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        fraud_count = y.sum()
        non_fraud_count = len(y) - fraud_count
        target_fraud_count = int(non_fraud_count * self.ratio)
        
        adasyn = ADASYN(
            sampling_strategy={1: target_fraud_count},
            random_state=self.random_state
        )
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        # Convert back to DataFrame if original was DataFrame
        if columns is not None:
            X_resampled = pd.DataFrame(X_resampled, columns=columns)
        
        return X_resampled, y_resampled
    
    def get_name(self):
        return f"ADASYN_{self.ratio}:1"


class BorderlineSMOTESampler(Sampler):
    """Borderline SMOTE sampling"""
    
    def __init__(self, ratio, random_state, k_neighbors=5):
        self.ratio = ratio
        self.random_state = random_state
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X, y):
        # Store column names if X is a DataFrame
        columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        fraud_count = y.sum()
        non_fraud_count = len(y) - fraud_count
        target_fraud_count = int(non_fraud_count * self.ratio)
        
        bsmote = BorderlineSMOTE(
            sampling_strategy={1: target_fraud_count},
            random_state=self.random_state,
            k_neighbors=self.k_neighbors
        )
        X_resampled, y_resampled = bsmote.fit_resample(X, y)
        
        if columns is not None:
            X_resampled = pd.DataFrame(X_resampled, columns=columns)
        
        return X_resampled, y_resampled
    
    def get_name(self):
        return f"BorderlineSMOTE_{self.ratio}:1"


class NoSampler(Sampler):
    """No sampling - pass through"""
    
    def fit_resample(self, X, y):
        return X, y
    
    def get_name(self):
        return "None"


class LightSMOTEENNSampler(Sampler):
    """SMOTE to 1:ratio then clean with ENN"""
    def __init__(self, ratio, random_state, smote_k=5, enn_k=3):
        self.ratio = ratio          # fraud : non-fraud after SMOTE (e.g. 0.1 -> 1:10)
        self.random_state = random_state
        self.smote_k = smote_k
        self.enn_k = enn_k

    def fit_resample(self, X, y):
        columns = X.columns if isinstance(X, pd.DataFrame) else None
        fraud = y.sum()
        non_fraud = len(y) - fraud
        target_fraud = int(non_fraud * self.ratio)

        # SMOTE ste
        smote = SMOTE(
            sampling_strategy={1: target_fraud},
            k_neighbors=min(self.smote_k, fraud-1),
            random_state=self.random_state
        )
        X_s, y_s = smote.fit_resample(X, y)

        # ENN cleaning step
        enn = EditedNearestNeighbours(
            n_neighbors=self.enn_k,
            kind_sel='all'          # remove samples whose majority neighbours disagree
        )
        X_res, y_res = enn.fit_resample(X_s, y_s)

        if columns is not None:
            X_res = pd.DataFrame(X_res, columns=columns)
        return X_res, y_res

    def get_name(self):
        return f"LightSMOTEENN_{self.ratio:.2f}:1"
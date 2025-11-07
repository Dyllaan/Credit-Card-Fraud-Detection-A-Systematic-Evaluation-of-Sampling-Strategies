import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class FeatureEngineer:
    """
    Feature engineering pipeline with caching for exact reproducibility.
    """
    def __init__(self, cache_manager, random_state=42):
        self.cache_manager = cache_manager
        self.random_state = random_state
        self.baseline_importance_ = None
        self.engineered_importance_ = None
        self.final_importance_ = None
        self.kept_features_ = None
        self.top_features_used_ = None
    
    def _generate_cache_key(self, X_train, y_train, top_n_engineer, prune_thresh):
        """
        Generate unique cache key from data and parameters.
        Uses data shape, column names, hash of first/last rows, and parameters.
        """
        # Data fingerprint
        data_info = {
            'train_shape': X_train.shape,
            'columns': list(X_train.columns) if isinstance(X_train, pd.DataFrame) else list(range(X_train.shape[1])),
            'y_sum': float(y_train.sum()),
            'y_len': len(y_train),
            # Hash first and last few rows for data identity
            'data_hash': hashlib.md5(
                np.concatenate([
                    X_train.iloc[:5].values.flatten() if isinstance(X_train, pd.DataFrame) else X_train[:5].flatten(),
                    X_train.iloc[-5:].values.flatten() if isinstance(X_train, pd.DataFrame) else X_train[-5:].flatten()
                ]).tobytes()
            ).hexdigest(),
            'top_n_engineer': top_n_engineer,
            'prune_thresh': prune_thresh,
            'random_state': self.random_state
        }
        
        # Create hash from all info
        key_str = json.dumps(data_info, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    
    def analyze_features(self, X_train, y_train, X_test, y_test, top_n=20):
        """
        Train 2 models, get normalized importance, print results.
        """
        # Convert to DataFrame if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        feature_names = X_train.columns.tolist()
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        print("Training models...")
        
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=self.random_state,
            eval_metric='aucpr'
        )
        xgb.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        xgb_imp = xgb.feature_importances_ / xgb.feature_importances_.sum()
        
        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=self.random_state, verbose=-1
        )
        lgbm.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)])
        lgbm_imp = lgbm.feature_importances_ / lgbm.feature_importances_.sum()

        # Create results
        importance_df = pd.DataFrame({
            'XGBoost': xgb_imp,
            'LightGBM': lgbm_imp,
        }, index=feature_names)
        importance_df['Mean'] = importance_df.mean(axis=1)
        importance_df['Std'] = importance_df.std(axis=1)
        importance_df = importance_df.sort_values('Mean', ascending=False)
        
        # Print results
        self._print_importance_table(importance_df, top_n)
        
        return importance_df
    
    def _print_importance_table(self, importance_df, top_n):
        """Print feature importance table"""
        print(f"Top {top_n} Features")
        print(f"{'#':<4}{'Feature':<30}{'Mean':<12}{'XGB':<12}{'LGBM':<12}{'Std':<12}")
        for i, (feat, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            print(f"{i:<4}{feat:<30}{row['Mean']:.6f}    {row['XGBoost']:.6f}    "
                  f"{row['LightGBM']:.6f}    {row['Std']:.6f}")
        
        print("Worst 10 Features")
        print(f"{'#':<4}{'Feature':<30}{'Mean':<12}{'XGB':<12}{'LGBM':<12}{'Std':<12}")
        total = len(importance_df)
        for i, (feat, row) in enumerate(importance_df.tail(10).iterrows(), total - 9):
            print(f"{i:<4}{feat:<30}{row['Mean']:.6f}    {row['XGBoost']:.6f}    "
                  f"{row['LightGBM']:.6f}    {row['Std']:.6f}")
    
    def engineer_features(self, X_train, X_test, top_features):
        """Create interactions, polynomials, ratios from top features"""
        print(f"\nEngineering from: {top_features}")
        
        X_train_eng = X_train.copy()
        X_test_eng = X_test.copy()
        
        count = 0
        
        # Pairwise interactions
        count += self._add_pairwise_interactions(X_train_eng, X_test_eng, top_features)
        
        # Squared interactions (top 3)
        count += self._add_squared_interactions(X_train_eng, X_test_eng, top_features[:3])
        
        # Polynomials (top 3)
        count += self._add_polynomials(X_train_eng, X_test_eng, top_features[:3])
        
        # Ratios (top 3)
        count += self._add_ratios(X_train_eng, X_test_eng, top_features[:3])
        
        # Amount features (if exists)
        if 'Amount' in X_train_eng.columns and 'Amount' in top_features[:7]:
            count += self._add_amount_features(X_train_eng, X_test_eng, top_features[:5])
        
        # Time features (if exists)
        if 'Time' in X_train_eng.columns:
            count += self._add_time_features(X_train_eng, X_test_eng)
        
        # Aggregations (top 5)
        count += self._add_aggregations(X_train_eng, X_test_eng, top_features[:5])
        
        print(f"Created {count} new features → {len(X_train_eng.columns)} total")
        
        return X_train_eng, X_test_eng
    
    def _add_pairwise_interactions(self, X_train, X_test, top_features):
        """Pairwise multiplication interactions"""
        count = 0
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i+1:]:
                X_train[f"{f1}_{f2}"] = X_train[f1] * X_train[f2]
                X_test[f"{f1}_{f2}"] = X_test[f1] * X_test[f2]
                count += 1
        return count
    
    def _add_squared_interactions(self, X_train, X_test, top_features):
        """Squared interactions"""
        count = 0
        for f1 in top_features:
            for f2 in top_features:
                if f1 != f2:
                    X_train[f"{f1}_{f2}_sq"] = (X_train[f1] * X_train[f2]) ** 2
                    X_test[f"{f1}_{f2}_sq"] = (X_test[f1] * X_test[f2]) ** 2
                    count += 1
        return count
    
    def _add_polynomials(self, X_train, X_test, top_features):
        """Polynomial transformations"""
        count = 0
        for f in top_features:
            X_train[f"{f}_sq"] = X_train[f] ** 2
            X_test[f"{f}_sq"] = X_test[f] ** 2
            
            X_train[f"{f}_cube"] = X_train[f] ** 3
            X_test[f"{f}_cube"] = X_test[f] ** 3
            
            X_train[f"{f}_sqrt"] = np.sqrt(np.abs(X_train[f]))
            X_test[f"{f}_sqrt"] = np.sqrt(np.abs(X_test[f]))
            
            X_train[f"{f}_log"] = np.log1p(np.abs(X_train[f]))
            X_test[f"{f}_log"] = np.log1p(np.abs(X_test[f]))
            count += 4
        return count
    
    def _add_ratios(self, X_train, X_test, top_features):
        """Ratio features"""
        count = 0
        for f1 in top_features:
            for f2 in top_features:
                if f1 != f2:
                    X_train[f"{f1}_div_{f2}"] = X_train[f1] / (X_train[f2].abs() + 1e-6)
                    X_test[f"{f1}_div_{f2}"] = X_test[f1] / (X_test[f2].abs() + 1e-6)
                    count += 1
        return count
    
    def _add_amount_features(self, X_train, X_test, top_features):
        """Amount-based interactions"""
        count = 0
        for f in top_features:
            if f != 'Amount':
                X_train[f"Amount_{f}"] = X_train['Amount'] * X_train[f]
                X_test[f"Amount_{f}"] = X_test['Amount'] * X_test[f]
                
                X_train[f"Amount_log_{f}"] = np.log1p(X_train['Amount']) * X_train[f]
                X_test[f"Amount_log_{f}"] = np.log1p(X_test['Amount']) * X_test[f]
                count += 2
        return count
    
    def _add_time_features(self, X_train, X_test):
        """Time-based cyclical features."""
        X_train['Hour'] = (X_train['Time'] / 3600) % 24
        X_test['Hour'] = (X_test['Time'] / 3600) % 24
        
        X_train['Hour_sin'] = np.sin(2 * np.pi * X_train['Hour'] / 24)
        X_test['Hour_sin'] = np.sin(2 * np.pi * X_test['Hour'] / 24)
        
        X_train['Hour_cos'] = np.cos(2 * np.pi * X_train['Hour'] / 24)
        X_test['Hour_cos'] = np.cos(2 * np.pi * X_test['Hour'] / 24)
        return 3
    
    def _add_aggregations(self, X_train, X_test, top_features):
        """Aggregation features across top features."""
        top_cols = [f for f in top_features if f in X_train.columns]
        
        X_train['Top5_mean'] = X_train[top_cols].mean(axis=1)
        X_test['Top5_mean'] = X_test[top_cols].mean(axis=1)
        
        X_train['Top5_std'] = X_train[top_cols].std(axis=1)
        X_test['Top5_std'] = X_test[top_cols].std(axis=1)
        
        X_train['Top5_max'] = X_train[top_cols].max(axis=1)
        X_test['Top5_max'] = X_test[top_cols].max(axis=1)
        
        X_train['Top5_min'] = X_train[top_cols].min(axis=1)
        X_test['Top5_min'] = X_test[top_cols].min(axis=1)
        
        return 4
    
    def prune_features(self, X_train, X_test, importance_df, threshold=0.005):
        """Remove features below importance threshold."""
        kept = importance_df[importance_df['Mean'] >= threshold].index.tolist()
        removed = importance_df[importance_df['Mean'] < threshold].index.tolist()
        
        print(f"\nPruning with threshold={threshold:.4f}")
        print(f"  Kept: {len(kept)}")
        print(f"  Removed: {len(removed)}")
        
        if removed:
            print(f"\nRemoved features:")
            for f in removed[:20]:
                imp = importance_df.loc[f, 'Mean']
                print(f"  - {f:<30} ({imp:.6f})")
            if len(removed) > 20:
                print(f"  ... and {len(removed) - 20} more")
        
        return X_train[kept], X_test[kept], kept
    
    def fit_transform(self, X_train, y_train, X_val, y_val,
                      top_n_engineer=5, prune_thresh=0.005, use_cache=True):
        """
        Complete workflow: baseline -> engineer -> prune -> final.
            use_cache: If True, try to load/save cached features
        """
        # Generate cache key
        cache_key = self._generate_cache_key(X_train, y_train, top_n_engineer, prune_thresh)
        
        # Try loading cache first
        if use_cache:
            cached = self.cache_manager.load_features(cache_key)
            if cached is not None:
                # Restore all cached results
                X_train_pruned = cached['X_train']
                X_test_pruned = cached['X_test']
                self.kept_features_ = cached['kept_features']
                self.baseline_importance_ = cached['baseline_importance']
                self.engineered_importance_ = cached['engineered_importance']
                self.final_importance_ = cached['final_importance']
                
                self.top_features_used_ = self.baseline_importance_.head(top_n_engineer).index.tolist()
                
                self._print_summary(from_cache=True, X_train_pruned=X_train_pruned)
                
                results = {
                    'baseline': self.baseline_importance_,
                    'engineered': self.engineered_importance_,
                    'final': self.final_importance_,
                    'kept_features': self.kept_features_,
                    'from_cache': True,
                    'cache_key': cache_key
                }
                
                return X_train_pruned, X_test_pruned, results
        
        # Full workflow if no cache
        print("Running Feature Engineering Pipeline")
        print("Running Baseline Analysis...")
        print("=" * 90)
        self.baseline_importance_ = self.analyze_features(
            X_train, y_train, X_val, y_val, top_n=20
        )
        
        print("Engineering Features...")
        top_feats = self.baseline_importance_.head(top_n_engineer).index.tolist()
        self.top_features_used_ = top_feats  # NEW: Store for transform()
        X_train_eng, X_val_eng = self.engineer_features(X_train, X_val, top_feats)
        
        print("Analyse Engineered Features...")
        self.engineered_importance_ = self.analyze_features(
            X_train_eng, y_train, X_val_eng, y_val, top_n=30
        )
        
        print("Prune Features with Lower Importance...")
        X_train_pruned, X_val_pruned, self.kept_features_ = self.prune_features(
            X_train_eng, X_val_eng, self.engineered_importance_, threshold=prune_thresh
        )
        
        print("Final Analysis of Pruned Features...")
        self.final_importance_ = self.analyze_features(
            X_train_pruned, y_train, X_test_pruned, y_val, top_n=25
        )
        
        # Save to cache
        if use_cache:
            self.cache_manager.save_features(
                cache_key, X_train_pruned, X_test_pruned, self.kept_features_,
                self.baseline_importance_, self.engineered_importance_, self.final_importance_, self.random_state
            )
        
        self._print_summary(
            from_cache=False,
            X_train=X_train,
            X_train_eng=X_train_eng,
            X_train_pruned=X_train_pruned
        )
        
        results = {
            'baseline': self.baseline_importance_,
            'engineered': self.engineered_importance_,
            'final': self.final_importance_,
            'kept_features': self.kept_features_,
            'from_cache': False,
            'cache_key': cache_key
        }
        
        return X_train_pruned, X_test_pruned, results
    
    def transform(self, X):
        """
        Apply learned feature engineering pipeline to new data (e.g: test set)
        """
        if self.kept_features_ is None or self.top_features_used_ is None:
            raise ValueError("Must call fit_transform() before transform()")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        print(f"\nApplying learned transformations to new data...")
        
        # Engineer features using the same top features
        X_eng, _ = self.engineer_features(X, X.copy(), self.top_features_used_)
        
        # Keep only the features that survived pruning
        X_transformed = X_eng[self.kept_features_]
        
        print(f"Transformed: {X.shape[0]} samples × {len(self.kept_features_)} features")
        
        return X_transformed
    
    def _print_summary(self, from_cache, X_train_pruned, X_train=None, X_train_eng=None):
        """Print workflow summary."""
        if from_cache:
            print("Loaded from cache")
            print(f"  Features: {len(X_train_pruned.columns)}")
            print(f"  Top 5: {self.final_importance_.head(5).index.tolist()}")
        else:
            print("Summary of Feature Engineering")
            print(f"  Started:    {len(X_train.columns)} features")
            print(f"  Engineered: {len(X_train_eng.columns)} features")
            print(f"  Final:      {len(X_train_pruned.columns)} features")
            print(f"\n  Top 5 baseline: {self.baseline_importance_.head(5).index.tolist()}")
            print(f"  Top 5 final:    {self.final_importance_.head(5).index.tolist()}")
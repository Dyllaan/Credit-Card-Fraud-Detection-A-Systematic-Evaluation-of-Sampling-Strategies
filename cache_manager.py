import pickle
import numpy as np
from pathlib import Path
import json

class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_dir = self.cache_dir / "features"
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, name):
        """Save model to cache"""
        filepath = self.cache_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Cached: {name}")

    def load_model(self, name):
        """Load model from cache if exists"""
        filepath = self.cache_dir / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

    def save_predictions(self, predictions, name):
        """Save predictions to cache"""
        filepath = self.cache_dir / f"{name}_predictions.npy"
        np.save(filepath, predictions)

    def load_predictions(self, name):
        """Load predictions from cache if exists"""
        filepath = self.cache_dir / f"{name}_predictions.npy"
        if filepath.exists():
            return np.load(filepath)
        return None

    def save_study(self, study, name):
        """Save Optuna study"""
        filepath = self.cache_dir / f"{name}_study.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(study, f)

    def load_study(self, name):
        """Load Optuna study if exists"""
        filepath = self.cache_dir / f"{name}_study.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _get_cache_path(self, cache_key):
        """Get cache file path for a given key."""
        return self.feature_cache_dir / f'features_{cache_key}.pkl'

    def save_features(self, cache_key, X_train, X_test, kept_features, 
                   baseline_importance, engineered_importance, final_importance, random_state):
        """Save engineered features for reproducibility."""
        self.feature_cache_dir.mkdir(exist_ok=True)
        
        cache_data = {
            'X_train': X_train,
            'X_test': X_test,
            'kept_features': kept_features,
            'baseline_importance': baseline_importance,
            'engineered_importance': engineered_importance,
            'final_importance': final_importance,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'cache_key': cache_key,
            'random_state': random_state
        }
        
        cache_file = self._get_cache_path(cache_key)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata for easy inspection
        metadata = {
            'cache_key': cache_key,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'n_features': len(kept_features),
            'top_5_features': final_importance.head(5).index.tolist() if final_importance is not None else [],
            'random_state': random_state
        }
        
        metadata_file = self.feature_cache_dir / f'metadata_{cache_key}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFeatures successfully cached")
        print(f"  Cache key: {cache_key}")
        print(f"  Location:  {cache_file}")
        print(f"  Train:     {X_train.shape}")
        print(f"  Test:      {X_test.shape}")
        print(f"  Features:  {len(kept_features)}")

    def load_features(self, cache_key):
        """Load cached feature."""
        cache_file = self._get_cache_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"\nLoaded cached features")
            print(f"  Cache key: {cache_key}")
            print(f"  Location:  {cache_file}")
            print(f"  Train:     {cache_data['train_shape']}")
            print(f"  Test:      {cache_data['test_shape']}")
            print(f"  Features:  {len(cache_data['kept_features'])}")
            
            return cache_data
        
        except Exception as e:
            print(f"\n✗ Failed to load cache: {e}")
            return None
        
    def list_features(self):
        """List all cached feature sets."""
        if not self.feature_cache_dir.exists():
            print("No cache directory found")
            return
        
        metadata_files = sorted(self.feature_cache_dir.glob('metadata_*.json'))
        
        if not metadata_files:
            print("No caches found")
            return
        
        print("Cached Feature Sets:")
        
        for i, mf in enumerate(metadata_files, 1):
            with open(mf, 'r') as f:
                meta = json.load(f)
            
            print(f"\n{i}. Cache Key: {meta['cache_key']}")
            print(f"   Train shape: {meta['train_shape']}")
            print(f"   Test shape:  {meta['test_shape']}")
            print(f"   Features:    {meta['n_features']}")
            print(f"   Top 5:       {meta['top_5_features']}")
            print(f"   Random seed: {meta['random_state']}")
        
        print(f"\n{'='*80}")

    def clear_features(self, cache_key=None):
        """Delete cached features. If cache_key=None, deletes all caches."""
        if cache_key:
            cache_file = self._get_cache_path(cache_key)
            metadata_file = self.feature_cache_dir / f'metadata_{cache_key}.json'
            
            deleted = False
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
            if metadata_file.exists():
                metadata_file.unlink()
                
            if deleted:
                print(f"Cache cleared: {cache_key}")
            else:
                print(f"No cache found: {cache_key}")
        else:
            # Clear all caches
            if self.feature_cache_dir.exists():
                cache_files = list(self.feature_cache_dir.glob('features_*.pkl'))
                metadata_files = list(self.feature_cache_dir.glob('metadata_*.json'))

                for f in cache_files + metadata_files:
                    f.unlink()
                
                print(f"Cleared {len(cache_files)} cache(s)")
            else:
                print("No cache directory found")
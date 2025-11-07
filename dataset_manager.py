import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetManager:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)

        print(f"Loaded dataset from {csv_file}")

        print("Dataset Shape:", self.df.shape)
        print("Dataset Shape:", self.df.shape)
        print("\nColumn Names:")
        print(self.df.columns.tolist())
        print("\nClass Distribution:")
        print(self.df['Class'].value_counts())
        print(f"\nFraud Rate: {self.df['Class'].mean()*100:.3f}%")
        print(f"Imbalance Ratio: 1:{int((len(self.df) - self.df['Class'].sum()) / self.df['Class'].sum())}")
    
    def create_train_test_split(self, test_size, random_state):
        # Train-test split with stratification
        X_temp = self.df.drop('Class', axis=1)
        y = self.df['Class']

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_temp, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("Train Set:")
        print(f"  Total: {len(X_train_raw):,}")
        print(f"  Frauds: {y_train.sum():,}")
        print(f"  Fraud rate: {y_train.mean()*100:.3f}%")

        print("\nTest Set:")
        print(f"  Total: {len(X_test_raw):,}")
        print(f"  Frauds: {y_test.sum():,}")
        print(f"  Fraud rate: {y_test.mean()*100:.3f}%")

        print("\nStratification preserved the class distribution")

        return X_train_raw, X_test_raw, y_train, y_test
    
    """
    Scale engineered features using StandardScaler i would do the engineering here, maybe i will in future 
    but as its dataset specific and this is designed to generalised
    """
    def scale_features(self, X_train_eng, X_test_eng):
        print("Scaling features with StandardScaler...")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)

        print(f"  Scaled {X_train_scaled.shape[1]} features")
        print(f"  Train shape: {X_train_scaled.shape}")
        print(f"  Test shape: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled
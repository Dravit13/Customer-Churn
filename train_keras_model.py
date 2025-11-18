"""
Train Keras/TensorFlow Model for Customer Churn Prediction
Creates a model.h5 file with a neural network model
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("✓ TensorFlow imported successfully!")
except ImportError:
    print("⚠ TensorFlow not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "tensorflow>=2.13.0"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class KerasChurnPredictor:
    """Keras/TensorFlow Model for Customer Churn Prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        self.n_components = None
        
    def download_and_load_data(self):
        """Download and load dataset"""
        print("=" * 60)
        print("STEP 1: Downloading and Loading Dataset")
        print("=" * 60)
        
        # Download dataset
        try:
            path = kagglehub.dataset_download("ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
            
            # Find data files
            data_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.csv', '.xlsx', '.xls')):
                        data_files.append(os.path.join(root, file))
            
            if not data_files:
                raise FileNotFoundError("No data files found")
            
            file_path = data_files[0]
            print(f"✓ Dataset found: {file_path}")
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                xl_file = pd.ExcelFile(file_path)
                sheet_names = xl_file.sheet_names
                
                # Find data sheet
                data_sheet = None
                for sheet in sheet_names:
                    sheet_lower = sheet.lower()
                    if 'dict' not in sheet_lower and 'meta' not in sheet_lower:
                        test_df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                        if len(test_df.columns) > 2:
                            data_sheet = sheet
                            break
                
                if data_sheet:
                    df = pd.read_excel(file_path, sheet_name=data_sheet)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            
            print(f"✓ Data loaded: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        print("\n" + "=" * 60)
        print("STEP 2: Data Preprocessing")
        print("=" * 60)
        
        df = df.copy()
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Identify target variable
        target_candidates = ['churn', 'Churn', 'Churned', 'churned', 'is_churn']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Could not identify target variable")
        
        print(f"✓ Target variable: {target_col}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
        
        print(f"✓ Preprocessing complete: {X.shape}")
        return X, y
    
    def handle_imbalance(self, X, y):
        """Handle imbalanced dataset"""
        print("\n" + "=" * 60)
        print("STEP 3: Handling Imbalanced Dataset")
        print("=" * 60)
        
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = min(counts) / max(counts)
        
        if imbalance_ratio < 0.5:
            print("⚠ Dataset imbalanced. Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"✓ Dataset balanced: {X_resampled.shape}")
            return X_resampled, y_resampled
        else:
            print("✓ Dataset is balanced")
            return X, y
    
    def split_and_scale(self, X, y, use_pca=True):
        """Split data, scale features, and apply PCA"""
        print("\n" + "=" * 60)
        print("STEP 4: Train-Test Split, Scaling & PCA")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features scaled")
        
        # Apply PCA
        if use_pca and X.shape[1] > 2:
            print(f"\nApplying PCA (95% variance)...")
            self.pca = PCA(n_components=0.95, random_state=42)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            
            self.n_components = self.pca.n_components_
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"✓ PCA applied: {X.shape[1]} → {self.n_components} features")
            print(f"  Variance retained: {explained_variance*100:.2f}%")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim):
        """Build Keras neural network model"""
        print("\n" + "=" * 60)
        print("STEP 5: Building Keras Model")
        print("=" * 60)
        
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✓ Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the Keras model"""
        print("\n" + "=" * 60)
        print("STEP 6: Training Keras Model")
        print("=" * 60)
        
        input_dim = X_train.shape[1]
        self.model = self.build_model(input_dim)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_train_pred = (self.model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
        y_test_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        y_test_proba = self.model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        overfitting_gap = train_acc - test_acc
        
        print(f"\n✓ Training Complete!")
        print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        return {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'overfitting_gap': float(overfitting_gap),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
    
    def save_model(self, metrics):
        """Save model as model.h5 and preprocessing components"""
        print("\n" + "=" * 60)
        print("STEP 7: Saving Model")
        print("=" * 60)
        
        # Save Keras model as .h5
        model_path = 'model.h5'
        self.model.save(model_path)
        print(f"✓ Keras model saved: {model_path}")
        
        # Save preprocessing components
        joblib.dump(self.scaler, 'scaler.pkl')
        print("✓ Scaler saved: scaler.pkl")
        
        if self.pca:
            joblib.dump(self.pca, 'pca.pkl')
            print("✓ PCA saved: pca.pkl")
        
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("✓ Label encoders saved: label_encoders.pkl")
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'n_components': int(self.n_components) if self.n_components else None,
            'use_pca': self.pca is not None,
            'metrics': metrics,
            'model_type': 'keras_neural_network',
            'input_shape': (None, self.n_components if self.n_components else len(self.feature_names))
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✓ Metadata saved: model_metadata.json")
        
        print(f"\n✓ All files saved successfully!")
        print(f"  - model.h5 (Keras model)")
        print(f"  - scaler.pkl")
        print(f"  - pca.pkl")
        print(f"  - label_encoders.pkl")
        print(f"  - model_metadata.json")


def main():
    """Main function"""
    predictor = KerasChurnPredictor()
    
    # Step 1: Download and load data
    df = predictor.download_and_load_data()
    
    # Step 2: Preprocess
    X, y = predictor.preprocess_data(df)
    
    # Step 3: Handle imbalance
    X_balanced, y_balanced = predictor.handle_imbalance(X, y)
    
    # Step 4: Split and scale
    X_train, X_test, y_train, y_test = predictor.split_and_scale(X_balanced, y_balanced)
    
    # Step 5-6: Build and train model
    metrics = predictor.train_model(X_train, X_test, y_train, y_test)
    
    # Step 7: Save model
    predictor.save_model(metrics)
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved as: model.h5")
    print(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Overfitting Gap: {metrics['overfitting_gap']:.4f}")


if __name__ == "__main__":
    main()


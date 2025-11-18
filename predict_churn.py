"""
Prediction Function for Customer Churn Model
Loads model.h5 and provides prediction function
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    print("✓ TensorFlow imported successfully!")
except ImportError:
    print("⚠ TensorFlow not installed. Please install: pip install tensorflow")
    raise


class ChurnPredictor:
    """Load and use the trained Keras model for predictions"""
    
    def __init__(self, model_path='model.h5'):
        """Initialize predictor by loading model and preprocessing components"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_encoders = {}
        self.feature_names = None
        self.metadata = None
        
        self.load_model()
    
    def load_model(self):
        """Load model.h5 and all preprocessing components"""
        print("Loading model and preprocessing components...")
        
        # Load Keras model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✓ Model loaded: {self.model_path}")
        
        # Load scaler
        if os.path.exists('scaler.pkl'):
            self.scaler = joblib.load('scaler.pkl')
            print("✓ Scaler loaded")
        else:
            raise FileNotFoundError("scaler.pkl not found")
        
        # Load PCA (optional)
        if os.path.exists('pca.pkl'):
            self.pca = joblib.load('pca.pkl')
            print("✓ PCA loaded")
        
        # Load label encoders
        if os.path.exists('label_encoders.pkl'):
            self.label_encoders = joblib.load('label_encoders.pkl')
            print("✓ Label encoders loaded")
        
        # Load metadata
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('feature_names', [])
            print("✓ Metadata loaded")
        else:
            print("⚠ Metadata not found, using default feature names")
            # Default feature names (update based on your dataset)
            self.feature_names = [
                'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
                'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                'DaySinceLastOrder', 'CashbackAmount'
            ]
        
        print("✓ All components loaded successfully!\n")
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        
        Args:
            input_data: dict or pandas DataFrame with feature values
            
        Returns:
            numpy array: Preprocessed features ready for model prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            # Ensure all features are present
            processed_dict = {}
            for feature in self.feature_names:
                if feature in input_data:
                    processed_dict[feature] = input_data[feature]
                else:
                    # Use default value (0 or mode)
                    processed_dict[feature] = 0.0
            
            # Create DataFrame in correct order
            df = pd.DataFrame([processed_dict])[self.feature_names]
        elif isinstance(input_data, pd.DataFrame):
            # Ensure correct column order
            df = input_data[self.feature_names].copy()
        else:
            raise ValueError("input_data must be dict or pandas DataFrame")
        
        # Encode categorical features
        for col in df.columns:
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                try:
                    # Try to transform
                    df[col] = encoder.transform(df[col].astype(str))
                except (ValueError, KeyError):
                    # Handle unseen categories - use 0 or most common
                    df[col] = 0
        
        # Convert to numpy array
        features = df.values.astype(float)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply PCA if available
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        return features_scaled
    
    def predict(self, input_data):
        """
        Predict churn for input data
        
        Args:
            input_data: dict or pandas DataFrame with feature values
                Example:
                {
                    'Tenure': 12,
                    'PreferredLoginDevice': 'Mobile Phone',
                    'CityTier': 3,
                    'WarehouseToHome': 10,
                    'PreferredPaymentMode': 'Credit Card',
                    'Gender': 'Male',
                    'HourSpendOnApp': 3.5,
                    'NumberOfDeviceRegistered': 2,
                    'PreferedOrderCat': 'Laptop & Accessory',
                    'SatisfactionScore': 3,
                    'MaritalStatus': 'Single',
                    'NumberOfAddress': 2,
                    'Complain': 0,
                    'OrderAmountHikeFromlastYear': 15,
                    'CouponUsed': 2,
                    'OrderCount': 5,
                    'DaySinceLastOrder': 10,
                    'CashbackAmount': 150.5
                }
        
        Returns:
            dict: Prediction results with churn probability and prediction
                {
                    'churn': 0 or 1,
                    'churn_probability': float (0-1),
                    'no_churn_probability': float (0-1)
                }
        """
        # Preprocess input
        processed_features = self.preprocess_input(input_data)
        
        # Make prediction
        probability = self.model.predict(processed_features, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'churn': int(prediction),
            'churn_probability': float(probability),
            'no_churn_probability': float(1 - probability)
        }
    
    def predict_batch(self, input_data_list):
        """
        Predict churn for multiple inputs
        
        Args:
            input_data_list: list of dicts or pandas DataFrame with multiple rows
        
        Returns:
            list: List of prediction results
        """
        if isinstance(input_data_list, list):
            # Process each input
            results = []
            for input_data in input_data_list:
                results.append(self.predict(input_data))
            return results
        elif isinstance(input_data_list, pd.DataFrame):
            # Process entire DataFrame
            processed_features = self.preprocess_input(input_data_list)
            probabilities = self.model.predict(processed_features, verbose=0).flatten()
            predictions = (probabilities > 0.5).astype(int)
            
            results = []
            for prob, pred in zip(probabilities, predictions):
                results.append({
                    'churn': int(pred),
                    'churn_probability': float(prob),
                    'no_churn_probability': float(1 - prob)
                })
            return results
        else:
            raise ValueError("input_data_list must be list or pandas DataFrame")


# Convenience function for easy use
def predict_churn(input_data, model_path='model.h5'):
    """
    Convenience function to predict churn
    
    Args:
        input_data: dict or pandas DataFrame with feature values
        model_path: path to model.h5 file (default: 'model.h5')
    
    Returns:
        dict: Prediction results
    """
    predictor = ChurnPredictor(model_path)
    return predictor.predict(input_data)


# Example usage
if __name__ == "__main__":
    # Example input data
    example_input = {
        'Tenure': 12,
        'PreferredLoginDevice': 'Mobile Phone',
        'CityTier': 3,
        'WarehouseToHome': 10,
        'PreferredPaymentMode': 'Credit Card',
        'Gender': 'Male',
        'HourSpendOnApp': 3.5,
        'NumberOfDeviceRegistered': 2,
        'PreferedOrderCat': 'Laptop & Accessory',
        'SatisfactionScore': 3,
        'MaritalStatus': 'Single',
        'NumberOfAddress': 2,
        'Complain': 0,
        'OrderAmountHikeFromlastYear': 15,
        'CouponUsed': 2,
        'OrderCount': 5,
        'DaySinceLastOrder': 10,
        'CashbackAmount': 150.5
    }
    
    try:
        # Initialize predictor
        predictor = ChurnPredictor('model.h5')
        
        # Make prediction
        result = predictor.predict(example_input)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Churn Prediction: {'YES' if result['churn'] == 1 else 'NO'}")
        print(f"Churn Probability: {result['churn_probability']:.4f} ({result['churn_probability']*100:.2f}%)")
        print(f"No Churn Probability: {result['no_churn_probability']:.4f} ({result['no_churn_probability']*100:.2f}%)")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python train_keras_model.py' first to create model.h5")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


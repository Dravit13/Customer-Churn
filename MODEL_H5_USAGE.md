# Model.h5 Usage Guide

This guide explains how to train and use the `model.h5` Keras/TensorFlow model for customer churn prediction.

## 📋 Files Created

1. **`train_keras_model.py`** - Trains a Keras neural network model and saves it as `model.h5`
2. **`predict_churn.py`** - Loads `model.h5` and provides prediction functions
3. **`model.h5`** - The trained Keras model (created after training)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_keras_model.py
```

This will:
- Download the dataset from Kaggle
- Preprocess the data
- Train a Keras neural network model
- Save the model as `model.h5`
- Save preprocessing components (scaler, PCA, encoders)

### Step 3: Use the Model for Predictions

#### Option A: Using the Predictor Class

```python
from predict_churn import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor('model.h5')

# Prepare input data
input_data = {
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

# Make prediction
result = predictor.predict(input_data)

print(f"Churn: {'YES' if result['churn'] == 1 else 'NO'}")
print(f"Churn Probability: {result['churn_probability']:.2%}")
```

#### Option B: Using the Convenience Function

```python
from predict_churn import predict_churn

input_data = {
    'Tenure': 12,
    'PreferredLoginDevice': 'Mobile Phone',
    # ... other features
}

result = predict_churn(input_data)
print(result)
```

#### Option C: Batch Predictions

```python
from predict_churn import ChurnPredictor
import pandas as pd

predictor = ChurnPredictor('model.h5')

# Multiple inputs as list
input_list = [
    {'Tenure': 12, 'PreferredLoginDevice': 'Mobile Phone', ...},
    {'Tenure': 6, 'PreferredLoginDevice': 'Computer', ...},
]

results = predictor.predict_batch(input_list)

# Or use DataFrame
df = pd.DataFrame(input_list)
results = predictor.predict_batch(df)
```

## 📊 Input Features

The model expects the following features (in any order):

- `Tenure` (numeric)
- `PreferredLoginDevice` (categorical)
- `CityTier` (numeric)
- `WarehouseToHome` (numeric)
- `PreferredPaymentMode` (categorical)
- `Gender` (categorical)
- `HourSpendOnApp` (numeric)
- `NumberOfDeviceRegistered` (numeric)
- `PreferedOrderCat` (categorical)
- `SatisfactionScore` (numeric)
- `MaritalStatus` (categorical)
- `NumberOfAddress` (numeric)
- `Complain` (numeric: 0 or 1)
- `OrderAmountHikeFromlastYear` (numeric)
- `CouponUsed` (numeric)
- `OrderCount` (numeric)
- `DaySinceLastOrder` (numeric)
- `CashbackAmount` (numeric)

## 📤 Output Format

The prediction function returns a dictionary:

```python
{
    'churn': 0 or 1,                    # Binary prediction
    'churn_probability': 0.0 to 1.0,    # Probability of churn
    'no_churn_probability': 0.0 to 1.0  # Probability of no churn
}
```

## 🔧 Model Architecture

The Keras model uses:
- **Input Layer**: Dense layer with 128 neurons (ReLU activation)
- **Hidden Layers**: 
  - Dense(64) with Dropout(0.3)
  - Dense(32) with Dropout(0.2)
- **Output Layer**: Dense(1) with Sigmoid activation (binary classification)
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Binary crossentropy
- **Regularization**: Dropout layers to prevent overfitting

## 📁 Files Generated

After training, the following files are created:

- `model.h5` - Keras model file
- `scaler.pkl` - Feature scaler
- `pca.pkl` - PCA transformer (if used)
- `label_encoders.pkl` - Label encoders for categorical features
- `model_metadata.json` - Model metadata and configuration

## ⚠️ Important Notes

1. **Model Training**: The model needs to be trained first using `train_keras_model.py`
2. **Preprocessing**: All preprocessing (scaling, encoding, PCA) is handled automatically
3. **Missing Features**: If a feature is missing, it defaults to 0.0
4. **Unseen Categories**: Unseen categorical values are encoded as 0

## 🧪 Example Script

Run the example:

```bash
python predict_churn.py
```

This will use example input data and show the prediction result.

## 📝 Integration Example

```python
# In your application
from predict_churn import ChurnPredictor

# Initialize once (can be reused)
predictor = ChurnPredictor('model.h5')

# Use in your application
def check_customer_churn(customer_data):
    result = predictor.predict(customer_data)
    
    if result['churn'] == 1:
        return {
            'status': 'HIGH_RISK',
            'probability': result['churn_probability'],
            'action': 'Send retention offer'
        }
    else:
        return {
            'status': 'LOW_RISK',
            'probability': result['churn_probability'],
            'action': 'Monitor customer'
        }
```

## 🐛 Troubleshooting

**Error: model.h5 not found**
- Run `python train_keras_model.py` first to create the model

**Error: TensorFlow not installed**
- Install: `pip install tensorflow>=2.13.0`

**Error: Feature mismatch**
- Ensure all required features are provided in the input data
- Check `model_metadata.json` for expected feature names


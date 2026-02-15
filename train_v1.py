"""
LithosGuard Pro - Training Pipeline
Generates the XGBoost model for seismic event classification.

Run this script to create: models/seismic_classifier.pkl
"""

import pandas as pd
import numpy as np
import os

print("🚀 LithosGuard Pro - ML Training Pipeline")
print("=" * 50)

# Step 1: Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('src', exist_ok=True)
print("✅ Directory structure verified")

# Step 2: Generate training dataset
def generate_training_set(rows=2000):
    """
    Generate synthetic training data for XGBoost classifier.
    
    Physics-based logic:
    - Pore Pressure and Displacement are correlated
    - Failure occurs when Displacement > 7mm
    """
    print(f"\n📊 Generating {rows} training samples...")
    
    # Independent variable: Pore Water Pressure (10-90 kPa)
    pressure = np.random.uniform(10, 90, rows)
    
    # Dependent variable: Displacement (physics-informed relationship)
    # Higher pressure → Higher displacement
    displacement = (pressure * 0.1) + np.random.normal(0, 1, rows)
    
    # Label: Failure classification (Binary)
    # Failure threshold: Displacement > 7mm
    label = (displacement > 7).astype(int)
    
    print(f"   - Features: Pore Pressure, Displacement")
    print(f"   - Failure cases: {np.sum(label)} / {rows} ({100*np.sum(label)/rows:.1f}%)")
    
    return pd.DataFrame({'Pressure': pressure, 'Displacement': displacement}), label

X_train, y_train = generate_training_set()

# Step 3: Train XGBoost Classifier
print("\n🧠 Training XGBoost Seismic Classifier...")

try:
    from xgboost import XGBClassifier
    import joblib
    
    # Initialize model with optimized hyperparameters
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate on training set (for verification)
    train_accuracy = model.score(X_train, y_train)
    print(f"   - Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Step 4: Save the model
    model_path = 'models/seismic_classifier.pkl'
    joblib.dump(model, model_path)
    
    print(f"\n✅ SUCCESS: Model saved to '{model_path}'")
    print(f"   - Model Type: XGBoost Classifier")
    print(f"   - Features: 2 (Pressure, Displacement)")
    print(f"   - Classes: 2 (Stable, Failure)")
    print(f"   - File Size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    print("\n" + "=" * 50)
    print("🎯 Your AI model is ready!")
    print("   Run 'streamlit run app.py' to launch the dashboard.")
    
except ImportError as e:
    print(f"\n❌ ERROR: Required package not found")
    print(f"   {e}")
    print("\n   Install dependencies:")
    print("   pip install xgboost scikit-learn joblib")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")

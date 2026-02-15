"""
LithosGuard Pro - Machine Learning Engine
Handles pre-trained AI models for seismic classification and visual analysis.
"""

import numpy as np
import os


class LithosML:
    """
    Machine Learning engine for LithosGuard Pro.
    Integrates XGBoost classifier and EfficientNet-B0 visual analysis.
    """
    
    def __init__(self, model_path='models/seismic_classifier.pkl'):
        """
        Initialize the ML engine.
        
        Args:
            model_path (str): Path to the trained XGBoost model
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Attempt to load model
        self._load_model()
    
    def _load_model(self):
        """Internal method to load the XGBoost model."""
        try:
            import joblib
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.model_loaded = True
                print(f"✅ Model loaded: {self.model_path}")
            else:
                print(f"⚠️ Model not found: {self.model_path}")
                print("   Run 'python train_v1.py' to generate the model.")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
    
    def predict_risk(self, pressure, displacement):
        """
        Predict failure risk using the trained XGBoost model.
        
        Args:
            pressure (float): Pore water pressure (kPa)
            displacement (float): Displacement (mm)
        
        Returns:
            tuple: (risk_label, probability)
        """
        if not self.model_loaded:
            return "Model Not Loaded", 0.0
        
        try:
            import pandas as pd
            # Prepare input
            X = pd.DataFrame({'Pressure': [pressure], 'Displacement': [displacement]})
            
            # Get prediction and probability
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            risk_label = "CRITICAL FAILURE" if prediction == 1 else "Stable"
            risk_prob = probability[1] if len(probability) > 1 else 0.5
            
            return risk_label, risk_prob
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return "Error", 0.0
    
    def classify_seismic_event(self, vibration_amplitude, dominant_frequency):
        """
        Classify seismic events using frequency analysis.
        
        Simulates XGBoost classification logic:
        - Low frequency (<60Hz): Heavy Machinery
        - High frequency (>1000Hz) + High amplitude: Rock Fracture
        
        Args:
            vibration_amplitude (float): Vibration amplitude (mm/s or g)
            dominant_frequency (float): Dominant frequency component (Hz)
        
        Returns:
            str: Classification label
        """
        if dominant_frequency < 60:
            return "Heavy Machinery (Noise)"
        elif dominant_frequency > 1000 and vibration_amplitude > 0.3:
            return "CRITICAL FRACTURE"
        elif dominant_frequency > 1000:
            return "Rock Micro-seismicity"
        else:
            return "Background Vibration"
    
    def analyze_crack_image(self, crack_intensity, confidence_threshold=0.95):
        """
        Simulate EfficientNet-B0 visual crack detection.
        
        In production, this would process actual images through a CNN.
        Here we simulate the output based on intensity metrics.
        
        Args:
            crack_intensity (float): Normalized crack intensity (0-1)
            confidence_threshold (float): Minimum confidence for detection
        
        Returns:
            tuple: (crack_width_mm, confidence_score)
        """
        # Base crack width
        base_width = 0.5  # mm
        
        # Exponential growth with intensity
        if crack_intensity > 0.7:
            growth = np.random.exponential(crack_intensity * 5)
            crack_width = base_width + growth
        else:
            crack_width = base_width + (crack_intensity * 0.5)
        
        # Simulated confidence (EfficientNet-B0 typically achieves >95% on clear images)
        confidence = min(0.98, 0.85 + (crack_intensity * 0.13))
        
        return round(crack_width, 2), round(confidence, 3)
    
    def get_model_info(self):
        """
        Get information about loaded models.
        
        Returns:
            dict: Model metadata
        """
        return {
            'xgboost_loaded': self.model_loaded,
            'model_path': self.model_path,
            'efficientnet_status': 'Simulated (Edge-Optimized)',
            'architecture': 'Multi-Modal (XGBoost + EfficientNet-B0)'
        }

"""
LithosGuard Pro - Physics Engine
Implements geotechnical failure criteria and predictive algorithms.
"""

import numpy as np


class GeotechPhysics:
    """
    Physics-based stability analysis using industry-standard methods.
    """
    
    def __init__(self):
        """Initialize the physics engine with default parameters."""
        self.model_version = "1.0.0"
    
    def calculate_fos(self, cohesion, friction_angle, normal_stress, pore_pressure, shear_stress):
        """
        Calculate Factor of Safety using Mohr-Coulomb Failure Criterion.
        
        Formula: FoS = (c' + (σ_n - u) × tan(φ')) / τ
        
        Args:
            cohesion (float): Effective cohesion (kPa)
            friction_angle (float): Internal friction angle (degrees)
            normal_stress (float): Normal stress on failure plane (kPa)
            pore_pressure (float): Pore water pressure (kPa)
            shear_stress (float): Driving shear stress (kPa)
        
        Returns:
            float: Factor of Safety (FoS)
        """
        # Calculate effective normal stress (Terzaghi's Principle)
        effective_normal_stress = max(normal_stress - pore_pressure, 0.1)
        
        # Convert friction angle to radians
        tan_phi = np.tan(np.radians(friction_angle))
        
        # Calculate shear strength
        shear_strength = cohesion + (effective_normal_stress * tan_phi)
        
        # Calculate Factor of Safety
        fos = shear_strength / max(shear_stress, 0.1)
        
        return fos
    
    def inverse_velocity(self, displacement_series, time_series=None):
        """
        Calculate Time-to-Failure using Fukuzono (1985) Inverse Velocity Method.
        
        The method is based on the empirical observation that 1/v → 0 implies failure.
        
        Args:
            displacement_series (array-like): Displacement measurements (mm)
            time_series (array-like, optional): Time points corresponding to displacements
        
        Returns:
            float: Inverse velocity (1/v), higher values indicate stability
        """
        displacement_array = np.array(displacement_series)
        
        if len(displacement_array) < 5:
            return 999.0  # Insufficient data - assume stable
        
        # Use last 10 points for velocity calculation
        recent_displacement = displacement_array[-10:]
        
        if time_series is not None:
            time_array = np.array(time_series)[-10:]
            # Calculate velocity using time-aware gradient
            velocity = np.gradient(recent_displacement, time_array)[-1]
        else:
            # Calculate velocity using simple gradient
            velocity = np.gradient(recent_displacement)[-1]
        
        # Avoid division by zero
        if abs(velocity) < 0.0001:
            return 99.0  # Nearly zero velocity - stable
        
        # Return inverse velocity
        return 1.0 / abs(velocity)
    
    def calculate_ttf(self, displacement_series, time_series=None, scaling_factor=0.5):
        """
        Estimate Time-to-Failure (TTF) in hours based on inverse velocity.
        
        Args:
            displacement_series (array-like): Displacement measurements (mm)
            time_series (array-like, optional): Time points
            scaling_factor (float): Calibration factor for time estimation
        
        Returns:
            float: Estimated time to failure (hours)
        """
        inv_v = self.inverse_velocity(displacement_series, time_series)
        
        # Map inverse velocity to time estimate
        ttf_hours = inv_v * scaling_factor
        
        return ttf_hours
    
    def calculate_shear_stress(self, displacement, base_stress=80.0, stress_increase_rate=2.5):
        """
        Calculate current shear stress based on displacement.
        
        Displacement increases effective shear stress due to strain-softening.
        
        Args:
            displacement (float): Current displacement (mm)
            base_stress (float): Base shear stress (kPa)
            stress_increase_rate (float): Stress increase per mm displacement
        
        Returns:
            float: Current shear stress (kPa)
        """
        return base_stress + (displacement * stress_increase_rate)

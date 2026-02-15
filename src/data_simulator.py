"""
LithosGuard Pro - Data Simulator
Generates GSI Bhukosh-standard geotechnical datasets.
"""

import numpy as np
import pandas as pd


def generate_gsi_dataset(rows=1000, scenario='monsoon'):
    """
    Generate synthetic geotechnical data following GSI Bhukosh metadata standards.
    
    GSI Bhukosh Fields:
    - Lithology: Rock type classification
    - RQD: Rock Quality Designation (0-100%)
    - Pore_Water_Pressure_kPa: Groundwater pressure
    - Displacement_mm: Slope movement
    - Raw_Acoustic_Emission_g: Vibration sensors
    
    Args:
        rows (int): Number of data points to generate
        scenario (str): 'monsoon', 'stable', or 'seismic'
    
    Returns:
        pd.DataFrame: GSI-compliant dataset
    """
    # Time series (24 hours)
    time_hrs = np.linspace(0, 24, rows)
    
    # GSI Metadata Fields
    lithology = ["Quartzite (Fractured)"] * rows
    rqd = np.random.randint(40, 50, rows)  # Rock Quality Designation
    
    # Scenario-specific generation
    if scenario == 'monsoon':
        data = _generate_monsoon_scenario(time_hrs, rows)
    elif scenario == 'seismic':
        data = _generate_seismic_scenario(time_hrs, rows)
    else:
        data = _generate_stable_scenario(time_hrs, rows)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'Time_Hrs': time_hrs,
        'Lithology': lithology,
        'RQD': rqd,
        'Pore_Water_Pressure_kPa': data['pressure'],
        'Raw_Acoustic_Emission_g': data['acoustic'],
        'Displacement_mm': data['displacement'],
        'Scenario_Intensity': data['intensity']
    })
    
    return df


def _generate_monsoon_scenario(time_hrs, rows):
    """
    Simulate monsoon-induced slope failure.
    
    Rain event starts at hour 15, peaks at hour 18.
    Pore pressure rises, triggering creep acceleration.
    """
    # Sigmoid function for rain intensity
    rain_event = 1 / (1 + np.exp(-0.5 * (time_hrs - 18)))
    
    # Pore Water Pressure (20 kPa baseline → 85 kPa peak)
    pressure = 20 + (rain_event * 65)
    
    # Acoustic Emissions
    # Low-freq truck noise (constant)
    truck_noise = 0.2 * np.sin(2 * np.pi * 0.5 * time_hrs)
    
    # High-freq fracture signals (appear when pressure > 60 kPa)
    fracture_signal = np.zeros(rows)
    fracture_signal[pressure > 60] = 0.05 * np.random.normal(0, 1, np.sum(pressure > 60))
    
    acoustic = truck_noise + fracture_signal + np.random.normal(0, 0.01, rows)
    
    # Displacement (exponential creep)
    displacement = 0.5 * np.exp(0.3 * (pressure / 10))
    
    return {
        'pressure': pressure,
        'acoustic': acoustic,
        'displacement': displacement,
        'intensity': rain_event
    }


def _generate_seismic_scenario(time_hrs, rows):
    """
    Simulate earthquake-triggered rockfall.
    
    P-wave arrival at hour 12, followed by S-wave.
    """
    # Base pressure
    pressure = 30 + np.random.normal(0, 2, rows)
    
    # Seismic spike at hour 12
    seismic_pulse = np.exp(-((time_hrs - 12) ** 2) / 2)
    acoustic = seismic_pulse * 2.0 + np.random.normal(0, 0.05, rows)
    
    # Displacement spike
    displacement = 1.0 + (seismic_pulse * 8)
    
    return {
        'pressure': pressure,
        'acoustic': acoustic,
        'displacement': displacement,
        'intensity': seismic_pulse
    }


def _generate_stable_scenario(time_hrs, rows):
    """
    Simulate stable operations (baseline noise only).
    """
    pressure = 25 + np.random.normal(0, 1, rows)
    acoustic = 0.02 * np.sin(2 * np.pi * 0.5 * time_hrs) + np.random.normal(0, 0.01, rows)
    displacement = np.linspace(0.5, 1.5, rows) + np.random.normal(0, 0.05, rows)
    
    return {
        'pressure': pressure,
        'acoustic': acoustic,
        'displacement': displacement,
        'intensity': np.zeros(rows)
    }


def perform_fft_analysis(signal_array, sampling_rate=1.0):
    """
    Perform Fast Fourier Transform to separate frequency components.
    
    Args:
        signal_array (array-like): Input signal
        sampling_rate (float): Sampling rate (Hz)
    
    Returns:
        tuple: (low_freq_energy, high_freq_energy)
    """
    signal = np.array(signal_array)
    
    # Simulated FFT energy calculation
    # In production, would use np.fft.fft(signal)
    
    # Low frequency energy (0-60 Hz) - Machinery noise
    low_freq_energy = np.mean(np.abs(signal)) * 0.8
    
    # High frequency energy (>1000 Hz) - Rock fracture
    high_freq_energy = np.var(signal) * 100
    
    return low_freq_energy, high_freq_energy

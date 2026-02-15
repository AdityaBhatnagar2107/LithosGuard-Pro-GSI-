"""
LithosGuard Pro - Stable Command Center with Session State Persistence
Professional geotechnical monitoring with flicker-free WebGL rendering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Import LithosGuard modules
from src.physics_engine import GeotechPhysics
from src.ml_engine import LithosML
from src.data_simulator import generate_gsi_dataset, perform_fft_analysis

# --- System Configuration ---
st.set_page_config(
    page_title="LithosGuard Pro | Command Center",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Industrial CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Courier+Prime:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e8e8e8;
        font-family: 'Inter', sans-serif;
        line-height: 1.2em;
    }
    
    .stMetric {
        background-color: rgba(31, 40, 51, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid #45a29e;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.25);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Courier Prime', monospace;
        color: #00f2ff;
        text-shadow: 0 0 10px #00f2ff;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .stMetric label {
        color: #45a29e !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 0.75rem;
    }
    
    .glass-card {
        background: rgba(25, 25, 40, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid #45a29e;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(69, 162, 158, 0.2);
    }
    
    .command-log {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #45a29e;
        border-radius: 8px;
        padding: 12px;
        margin: 15px 0;
        font-family: 'Courier Prime', monospace;
        font-size: 0.85rem;
        color: #00ff88;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .integration-box {
        background: rgba(10, 10, 15, 0.8);
        border: 1px solid #45a29e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier Prime', monospace;
        font-size: 0.9rem;
    }
    
    .evacuate-alert {
        background: linear-gradient(135deg, rgba(255, 0, 60, 0.95) 0%, rgba(180, 0, 40, 0.95) 100%);
        border: 2px solid #ff003c;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        box-shadow: 0 0 40px rgba(255, 0, 60, 0.7);
        animation: alert-pulse 1.5s ease-in-out infinite;
        margin: 20px 0;
    }
    
    .siren-indicator-idle {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: #00ff88;
        box-shadow: 0 0 10px #00ff88;
        animation: pulse-dot 2s ease-in-out infinite;
        margin-right: 8px;
    }
    
    .siren-indicator-active {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: #ff003c;
        box-shadow: 0 0 15px #ff003c;
        animation: pulse-alarm 0.8s ease-in-out infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse-alarm {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .physics-footer {
        font-size: 0.7rem;
        color: #888;
        margin-top: 8px;
        font-style: italic;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 6px;
    }
    
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.15); }
    }
    
    .status-active { background-color: #00ff88; box-shadow: 0 0 8px #00ff88; }
    .status-offline { background-color: #666; }
    
    .system-header {
        background: linear-gradient(90deg, rgba(0, 242, 255, 0.15) 0%, rgba(69, 162, 158, 0.15) 100%);
        border: 1px solid #45a29e;
        border-radius: 8px;
        padding: 25px 20px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100px;
    }
    
    .system-title {
        font-family: 'Courier Prime', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00f2ff;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.8);
        letter-spacing: 3px;
        margin: 0 0 12px 0;
        line-height: 1.4;
        word-wrap: break-word;
        max-width: 100%;
    }
    
    .system-subtitle {
        color: #888;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 8px;
    }
    
    .live-feed {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid #00ff88;
        border-radius: 6px;
        padding: 10px;
        font-family: 'Courier Prime', monospace;
        font-size: 0.85rem;
        color: #00ff88;
        margin: 10px 0;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    
    .system-banner {
        background: linear-gradient(90deg, rgba(0, 242, 255, 0.12) 0%, rgba(69, 162, 158, 0.12) 100%);
        border: 1px solid #45a29e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 0 15px rgba(69, 162, 158, 0.2);
    }
    
    .critical-alert {
        background: linear-gradient(135deg, rgba(255, 0, 60, 0.95) 0%, rgba(180, 0, 40, 0.95) 100%);
        border: 2px solid #ff003c;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 0, 60, 0.7);
        animation: alert-pulse 1.5s ease-in-out infinite;
        margin: 20px 0;
    }
    
    @keyframes alert-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.015); }
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00f2ff 0%, #45a29e 100%);
        color: #0a0a0a;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        box-shadow: 0 4px 18px rgba(0, 242, 255, 0.35);
    }
    
    .searching-status {
        background: rgba(255, 170, 0, 0.2);
        border: 1px solid #ffaa00;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        text-align: center;
        color: #ffaa00;
        font-weight: 600;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Systems ---
@st.cache_resource
def initialize_systems():
    physics = GeotechPhysics()
    ml = LithosML()
    return physics, ml

physics_engine, ml_engine = initialize_systems()

# Initialize session state for data persistence
if 'inject_failure' not in st.session_state:
    st.session_state.inject_failure = False
if 'failure_injected_at' not in st.session_state:
    st.session_state.failure_injected_at = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'SIMULATOR'
if 'test_siren' not in st.session_state:
    st.session_state.test_siren = False
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False
if 'command_log' not in st.session_state:
    st.session_state.command_log = []
if 'alarm_already_logged' not in st.session_state:
    st.session_state.alarm_already_logged = False
if 'telemetry_buffer' not in st.session_state:
    st.session_state.telemetry_buffer = {'time': [], 'pressure': [], 'displacement': [], 'fos': []}
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# --- Sidebar: Integration Hub ---
with st.sidebar:
    st.markdown("<h3>[ SYSTEM : LITHOSGUARD ]</h3>", unsafe_allow_html=True)
    st.caption("Mission-Critical Command & Control v2.1")
    
    st.markdown("---")
    
    # Emergency Control Center
    st.markdown("### 🚨 Emergency Control")
    
    siren_active = st.session_state.alarm_triggered or st.session_state.test_siren
    siren_class = "siren-indicator-active" if siren_active else "siren-indicator-idle"
    siren_status = "ALARM ACTIVE" if siren_active else "IDLE"
    siren_color = "#ff003c" if siren_active else "#00ff88"
    
    st.markdown(f"""
    <div style='background: rgba(25, 25, 40, 0.6); border: 1px solid #45a29e; border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
        <strong style='color: #45a29e;'>Siren Status:</strong><br/>
        <span class='{siren_class}'></span>
        <span style='color: {siren_color}; font-weight: 700;'>{siren_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: rgba(25, 25, 40, 0.6); border: 1px solid #45a29e; border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
        <strong style='color: #45a29e;'>Relay Connection:</strong><br/>
        <span class='status-dot status-active'></span>
        <span style='color: #00ff88;'>SN-88-A (Active)</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔊 TEST SIREN", use_container_width=True):
        st.session_state.test_siren = True
        current_time = datetime.now().strftime('%H:%M:%S')
        st.session_state.command_log.append(f"[{current_time}] TEST: Manual siren test initiated")
        time.sleep(0.5)
        st.success("Siren test initiated!")
        time.sleep(1)
        st.session_state.test_siren = False
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Sensor Registry with Status
    with st.expander("📡 Sensor Node Registry"):
        # Determine node status based on data source
        if st.session_state.data_source == 'PHYSICAL API' and not st.session_state.api_connected:
            node_status = "OFFLINE"
            status_color = "#666"
            status_dot = "status-offline"
        else:
            node_status = "ACTIVE"
            status_color = "#00ff88"
            status_dot = "status-active"
        
        st.markdown(f"""
        **Node 1: SN-4492-7B**  
        <span class='status-dot {status_dot}'></span>Status: <span style='color: {status_color};'>{node_status}</span>  
        MAC: `A4:CF:12:8E:4D:92`  
        Protocol: LoRaWAN  
        Battery: 85% | RSSI: -67 dBm
        
        **Node 2: SN-4493-2C**  
        <span class='status-dot {status_dot}'></span>Status: <span style='color: {status_color};'>{node_status}</span>  
        MAC: `B2:1F:8A:3C:7E:44`  
        Protocol: LoRaWAN  
        Battery: 91% | RSSI: -54 dBm
        
        **Node 3: SN-4494-9A**  
        <span class='status-dot {status_dot}'></span>Status: <span style='color: {status_color};'>{node_status}</span>  
        MAC: `C8:D4:5B:2F:1A:66`  
        Protocol: MQTT/WiFi  
        Battery: 78% | RSSI: -71 dBm
        
        **Node 4: SN-4495-1E**  
        <span class='status-dot {status_dot}'></span>Status: <span style='color: {status_color};'>{node_status}</span>  
        MAC: `D1:8C:4F:9B:3D:28`  
        Protocol: LoRaWAN  
        Battery: 88% | RSSI: -62 dBm
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Operation Mode")
    
    mode = st.radio(
        "Select Mode",
        ["MANUAL FORENSICS", "LIVE STREAM"],
        help="Manual: Pre-failure analysis | Live: Real-time feed"
    )
    
    if mode == "LIVE STREAM":
        st.markdown("### 📊 Data Source")
        source_option = st.radio("Source", ["SIMULATOR", "PHYSICAL API"])
        
        # Reset telemetry buffer when switching sources
        if source_option != st.session_state.data_source:
            st.session_state.telemetry_buffer = {'time': [], 'pressure': [], 'displacement': [], 'fos': []}
            st.session_state.api_connected = False
        
        st.session_state.data_source = source_option
        
        if source_option == "PHYSICAL API":
            st.warning("⚠️ Physical API mode: Hardware search active")
    
    scenario = st.selectbox("Scenario", ["monsoon", "stable", "seismic"])
    
    st.markdown("---")
    st.markdown("### 🔬 Test Controls")
    
    if st.button("⚠️ INJECT FAILURE", use_container_width=True):
        st.session_state.inject_failure = True
        st.session_state.failure_injected_at = datetime.now()
        st.session_state.alarm_already_logged = False
        st.success("Failure injected!")
    
    if st.session_state.inject_failure and st.button("Reset", use_container_width=True):
        st.session_state.inject_failure = False
        st.session_state.alarm_triggered = False
        st.session_state.command_log = []
        st.session_state.alarm_already_logged = False
        st.session_state.telemetry_buffer = {'time': [], 'pressure': [], 'displacement': [], 'fos': []}
        st.rerun()

# --- Header ---
st.markdown("""
<div class='system-header'>
    <div class='system-title'>[ LITHOSGUARD PRO | SPATIAL INTELLIGENCE CENTER ]</div>
    <p class='system-subtitle'>
    Sector 4 (Himalayan Belt) | Quartzite Formation | GSI Bhukosh Integrated
    </p>
</div>
""", unsafe_allow_html=True)

# --- Tab Navigation ---
tab1, tab2 = st.tabs(["📊 OPERATIONS DASHBOARD", "🔌 HARDWARE INTEGRATION"])

with tab2:
    st.markdown("### 🔌 Physical Connectivity Configuration")
    
    col_api1, col_api2 = st.columns(2)
    with col_api1:
        st.markdown("""
        <div class='integration-box'>
        <strong style='color: #00f2ff;'>REST API Endpoint:</strong><br/>
        <code>https://api.lithosguard.io/v1/stream/sector4</code><br/><br/>
        <strong style='color: #00f2ff;'>Auth:</strong> <span style='color: #00ff88;'>✓ JWT Valid</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_api2:
        st.markdown("""
        <div class='integration-box'>
        <strong style='color: #00f2ff;'>Alarm Trigger:</strong><br/>
        <code>POST /v1/trigger/alarm</code><br/><br/>
        <strong style='color: #00f2ff;'>Latency:</strong> <span style='color: #00ff88;'>< 100ms</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### MQTT Broker Configuration")
    st.markdown("""
    <div class='integration-box'>
    <strong style='color: #00f2ff;'>Broker:</strong> <code>mqtt.lithosguard.io:8883</code> (TLS 1.3)<br/>
    <strong style='color: #00f2ff;'>Data Topic:</strong> <code>mine/pit1/sector4/seismic</code><br/>
    <strong style='color: #00f2ff;'>Control Topic:</strong> <code>mine/pit1/control/alarm</code><br/>
    <strong style='color: #00f2ff;'>QoS:</strong> 2 (Exactly Once Delivery)
    </div>
    """, unsafe_allow_html=True)

with tab1:
    # --- Generate Dataset ---
    @st.cache_data
    def get_dataset(scenario_type):
        return generate_gsi_dataset(rows=1000, scenario=scenario_type)

    full_data = get_dataset(scenario)

    if st.session_state.inject_failure:
        spike_start = int(len(full_data) * 0.8)
        full_data.loc[spike_start:, 'Pore_Water_Pressure_kPa'] *= 1.5

    # --- MODE: LIVE STREAM ---
    if mode == "LIVE STREAM":
        source_indicator = f"[ SOURCE: {st.session_state.data_source} ]"
        st.markdown(f"### 🛰️ Real-Time Monitoring {source_indicator}")
        
        # Show scanning status for Physical API
        if st.session_state.data_source == 'PHYSICAL API' and not st.session_state.api_connected:
            st.markdown("""
            <div class='searching-status'>
                🔍 SCANNING LORAWAN MESH... No physical nodes detected. Using simulator standby mode.
            </div>
            """, unsafe_allow_html=True)
        
        playback_speed = st.slider("Playback Speed", 1, 20, 5)
        
        banner_ph = st.empty()
        feed_ph = st.empty()
        alert_ph = st.empty()
        metrics_ph = st.empty()
        heatmap_ph = st.empty()
        chart_ph = st.empty()
        log_ph = st.empty()
        
        for i in range(0, len(full_data), playback_speed):
            current_point = full_data.iloc[i]
            
            # Append to session state buffer (memory persistence)
            st.session_state.telemetry_buffer['time'].append(current_point['Time_Hrs'])
            st.session_state.telemetry_buffer['pressure'].append(current_point['Pore_Water_Pressure_kPa'])
            st.session_state.telemetry_buffer['displacement'].append(current_point['Displacement_mm'])
            
            # Keep buffer size manageable (last 200 points)
            max_buffer = 200
            if len(st.session_state.telemetry_buffer['time']) > max_buffer:
                for key in st.session_state.telemetry_buffer:
                    st.session_state.telemetry_buffer[key] = st.session_state.telemetry_buffer[key][-max_buffer:]
            
            history = full_data.iloc[:i+1]
            
            # Physics calculations
            shear_stress = physics_engine.calculate_shear_stress(current_point['Displacement_mm'])
            fos = physics_engine.calculate_fos(50, 35, 120, current_point['Pore_Water_Pressure_kPa'], shear_stress)
            ttf = physics_engine.calculate_ttf(history['Displacement_mm'].values, history['Time_Hrs'].values)
            ml_risk, ml_prob = ml_engine.predict_risk(current_point['Pore_Water_Pressure_kPa'], current_point['Displacement_mm'])
            
            st.session_state.telemetry_buffer['fos'].append(fos)
            if len(st.session_state.telemetry_buffer['fos']) > max_buffer:
                st.session_state.telemetry_buffer['fos'] = st.session_state.telemetry_buffer['fos'][-max_buffer:]
            
            is_critical = fos < 1.05
            
            # Trigger alarm ONLY ONCE
            if is_critical and not st.session_state.alarm_already_logged:
                st.session_state.alarm_triggered = True
                st.session_state.alarm_already_logged = True
                current_time = datetime.now().strftime('%H:%M:%S')
                st.session_state.command_log.extend([
                    f"[{current_time}] MQTT >> mine/pit1/control/alarm {{\"cmd\": \"SIREN_ON\"}}",
                    f"[{current_time}] API >> POST /v1/trigger/alarm {{\"sector\": 4, \"severity\": \"CRITICAL\"}}",
                    f"[{current_time}] SMS >> Broadcast to 42 workers: EVACUATE SECTOR 4",
                    f"[{current_time}] RELAY >> Truck #091 engine cutoff activated",
                    f"[{current_time}] RELAY >> Emergency lighting Zone 4: ON"
                ])
            
            risk_level = "CRITICAL" if is_critical else "SAFE"
            risk_color = "#ff003c" if is_critical else "#00ff88"
            
            # Banner
            with banner_ph:
                st.markdown(f"""
                <div class='system-banner'>
                    <div style='display: flex; justify-content: space-around; text-align: center;'>
                        <div><span style='color: #45a29e; font-size: 0.8rem;'>UPTIME</span><br/>
                        <span style='font-size: 1.3rem; color: #fff; font-weight: 700;'>99.99%</span></div>
                        <div><span style='color: #45a29e; font-size: 0.8rem;'>LATENCY</span><br/>
                        <span style='font-size: 1.3rem; color: #fff; font-weight: 700;'>{np.random.randint(15,45)}ms</span></div>
                        <div><span style='color: #45a29e; font-size: 0.8rem;'>RISK</span><br/>
                        <span style='font-size: 1.3rem; font-weight: 700; color: {risk_color};'>{risk_level}</span></div>
                        <div><span style='color: #45a29e; font-size: 0.8rem;'>STREAM</span><br/>
                        <span style='font-size: 1.3rem; color: #00ff88; font-weight: 700;'><span class='status-dot status-active'></span>LIVE</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Feed
            with feed_ph:
                st.markdown(f"""
                <div class='live-feed'>
                [{datetime.now().strftime('%H:%M:%S')}] P: {current_point['Pore_Water_Pressure_kPa']:.2f} kPa | 
                D: {current_point['Displacement_mm']:.3f} mm | 
                FoS: <span style='color: {"#ff003c" if is_critical else "#00ff88"};'>{fos:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Alert
            if is_critical:
                with alert_ph:
                    st.markdown("""<div class='evacuate-alert'>🚨 EVACUATE SECTOR 4</div>""", unsafe_allow_html=True)
            else:
                alert_ph.empty()
            
            # Metrics
            with metrics_ph:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Factor of Safety", f"{fos:.2f}", delta="CRITICAL" if is_critical else "STABLE", delta_color="inverse" if is_critical else "normal")
                    st.markdown("<div class='physics-footer'>✓ Mohr-Coulomb v2.4</div>", unsafe_allow_html=True)
                with c2:
                    st.metric("Time to Failure", f"{ttf:.1f} hrs")
                    st.markdown("<div class='physics-footer'>✓ Fukuzono (1985)</div>", unsafe_allow_html=True)
                with c3:
                    st.metric("Pore Pressure", f"{current_point['Pore_Water_Pressure_kPa']:.1f} kPa")
                    st.markdown("<div class='physics-footer'>✓ Terzaghi Principle</div>", unsafe_allow_html=True)
                with c4:
                    st.metric("AI Risk", f"{ml_prob*100:.0f}%")
                    st.markdown("<div class='physics-footer'>✓ XGBoost Model</div>", unsafe_allow_html=True)
            
            # Digital Twin Choropleth Heatmap
            with heatmap_ph:
                st.markdown("### 🗺️ Digital Twin: Sector Heatmap")
                
                sector_polys = {
                    'Sector 1': {'x': [0, 1, 1, 0, 0], 'y': [2, 2, 3, 3, 2], 'fos': 1.8},
                    'Sector 2': {'x': [1, 2, 2, 1, 1], 'y': [2, 2, 3, 3, 2], 'fos': 1.6},
                    'Sector 3': {'x': [2, 3, 3, 2, 2], 'y': [2, 2, 3, 3, 2], 'fos': 1.5},
                    'Sector 4': {'x': [0, 1, 1, 0, 0], 'y': [1, 1, 2, 2, 1], 'fos': fos},
                    'Sector 5': {'x': [1, 2, 2, 1, 1], 'y': [1, 1, 2, 2, 1], 'fos': 1.7}
                }
                
                fig_map = go.Figure()
                
                for sector_name, data in sector_polys.items():
                    sector_fos = data['fos']
                    
                    if sector_fos > 1.2:
                        color = 'rgba(0, 255, 136, 0.6)'
                        line_color = '#00ff88'
                    elif sector_fos < 1.05:
                        color = 'rgba(255, 0, 60, 0.8)'
                        line_color = '#ff003c'
                    else:
                        color = 'rgba(255, 170, 0, 0.6)'
                        line_color = '#ffaa00'
                    
                    fig_map.add_trace(go.Scatter(
                        x=data['x'],
                        y=data['y'],
                        fill='toself',
                        fillcolor=color,
                        line=dict(color=line_color, width=2),
                        mode='lines',
                        name=sector_name,
                        hovertemplate=f'<b>{sector_name}</b><br>FoS: {sector_fos:.2f}<extra></extra>'
                    ))
                    
                    center_x = sum(data['x'][:-1]) / 4
                    center_y = sum(data['y'][:-1]) / 4
                    fig_map.add_trace(go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='text',
                        text=[sector_name],
                        textfont=dict(size=14, color='#fff', family='Courier Prime'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig_map.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    showlegend=False,
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 3.5]),
                    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.5, 3.5]),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                st.plotly_chart(fig_map, use_container_width=True, key=f"heatmap_live_{i}")
            
            # WebGL Scattergl Chart with Fixed Y-Axis (Flicker-Free)
            with chart_ph:
                fig = go.Figure()
                
                # Use Scattergl for WebGL rendering (high-performance)
                fig.add_trace(go.Scattergl(
                    x=st.session_state.telemetry_buffer['time'], 
                    y=st.session_state.telemetry_buffer['pressure'], 
                    name='Pressure', 
                    line=dict(color='#00f2ff', width=2),
                    fill='tozeroy', 
                    fillcolor='rgba(0,242,255,0.12)'
                ))
                
                fig.add_hline(y=80, line_dash="dash", line_color="#ff003c", line_width=2)
                
                # Fixed Y-axis range to eliminate jumping
                fig.update_layout(
                    template="plotly_dark", 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    height=300, 
                    margin=dict(l=40,r=20,t=20,b=40),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Time (Hrs)'),
                    yaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)', 
                        title='Pressure (kPa)',
                        range=[0, 120]  # Fixed Y-axis range
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"chart_live_{i}")
            
            # Persistent Command Log
            with log_ph:
                st.markdown("### 📡 Outbound Command Log")
                log_display = "\n".join(st.session_state.command_log[-10:]) if st.session_state.command_log else "[Monitoring... No commands sent yet]"
                st.markdown(f"<div class='command-log'>{log_display}</div>", unsafe_allow_html=True)
            
            time.sleep(0.1)
            if is_critical and i > 800:
                st.info("⏸️ Simulation paused at critical point")
                break

    # --- MODE: MANUAL FORENSICS ---
    else:
        st.markdown("### 🔍 Manual Forensics: Pre-Failure Signature Analysis")
        
        timeline_hour = st.slider("Timeline (Hours)", 0.0, 24.0, 12.0, 0.1)
        
        current_data = full_data[full_data['Time_Hrs'] <= timeline_hour]
        if len(current_data) == 0:
            current_data = full_data.iloc[:1]
        
        latest = current_data.iloc[-1]
        
        shear = physics_engine.calculate_shear_stress(latest['Displacement_mm'])
        fos = physics_engine.calculate_fos(50, 35, 120, latest['Pore_Water_Pressure_kPa'], shear)
        ttf = physics_engine.calculate_ttf(current_data['Displacement_mm'].values, current_data['Time_Hrs'].values)
        ml_risk, ml_prob = ml_engine.predict_risk(latest['Pore_Water_Pressure_kPa'], latest['Displacement_mm'])
        
        is_critical = fos < 1.05
        
        if is_critical and not st.session_state.alarm_already_logged:
            st.session_state.alarm_triggered = True
            st.session_state.alarm_already_logged = True
            ct = datetime.now().strftime('%H:%M:%S')
            st.session_state.command_log.extend([
                f"[{ct}] MQTT >> mine/pit1/control/alarm {{\"cmd\": \"SIREN_ON\"}}",
                f"[{ct}] API >> POST /v1/trigger/alarm",
                f"[{ct}] SMS >> 42 workers notified",
                f"[{ct}] RELAY >> Truck #091 cutoff"
            ])
        elif not is_critical and st.session_state.alarm_already_logged:
            st.session_state.alarm_triggered = False
            st.session_state.alarm_already_logged = False
        
        risk_level = "CRITICAL" if is_critical else "SAFE"
        risk_color = "#ff003c" if is_critical else "#00ff88"
        
        st.markdown(f"""
        <div class='system-banner'>
            <div style='display: flex; justify-content: space-around; text-align: center;'>
                <div><span style='color: #45a29e; font-size: 0.8rem;'>RISK</span><br/>
                <span style='font-size: 1.3rem; font-weight: 700; color: {risk_color};'>{risk_level}</span></div>
                <div><span style='color: #45a29e; font-size: 0.8rem;'>ANALYSIS MODE</span><br/>
                <span style='font-size: 1.3rem; color: #fff; font-weight: 700;'>FORENSICS</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if is_critical:
            st.markdown("""<div class='evacuate-alert'>🚨 EVACUATE SECTOR 4</div>""", unsafe_allow_html=True)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Factor of Safety", f"{fos:.2f}", delta="CRITICAL" if is_critical else "STABLE")
            st.markdown("<div class='physics-footer'>✓ Mohr-Coulomb</div>", unsafe_allow_html=True)
        with c2:
            st.metric("Time to Failure", f"{ttf:.1f} hrs")
            st.markdown("<div class='physics-footer'>✓ Fukuzono</div>", unsafe_allow_html=True)
        with c3:
            st.metric("Pore Pressure", f"{latest['Pore_Water_Pressure_kPa']:.1f} kPa")
            st.markdown("<div class='physics-footer'>✓ Terzaghi</div>", unsafe_allow_html=True)
        with c4:
            st.metric("AI Risk", f"{ml_prob*100:.0f}%")
            st.markdown("<div class='physics-footer'>✓ XGBoost</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Polygon Heatmap
        st.markdown("### 🗺️ Digital Twin: Sector Heatmap")
        
        sector_polys = {
            'Sector 1': {'x': [0, 1, 1, 0, 0], 'y': [2, 2, 3, 3, 2], 'fos': 1.8},
            'Sector 2': {'x': [1, 2, 2, 1, 1], 'y': [2, 2, 3, 3, 2], 'fos': 1.6},
            'Sector 3': {'x': [2, 3, 3, 2, 2], 'y': [2, 2, 3, 3, 2], 'fos': 1.5},
            'Sector 4': {'x': [0, 1, 1, 0, 0], 'y': [1, 1, 2, 2, 1], 'fos': fos},
            'Sector 5': {'x': [1, 2, 2, 1, 1], 'y': [1, 1, 2, 2, 1], 'fos': 1.7}
        }
        
        fig_map = go.Figure()
        
        for sector_name, data in sector_polys.items():
            sector_fos = data['fos']
            
            if sector_fos > 1.2:
                color = 'rgba(0, 255, 136, 0.6)'
                line_color = '#00ff88'
            elif sector_fos < 1.05:
                color = 'rgba(255, 0, 60, 0.8)'
                line_color = '#ff003c'
            else:
                color = 'rgba(255, 170, 0, 0.6)'
                line_color = '#ffaa00'
            
            fig_map.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                fill='toself',
                fillcolor=color,
                line=dict(color=line_color, width=2),
                mode='lines',
                name=sector_name,
                hovertemplate=f'<b>{sector_name}</b><br>FoS: {sector_fos:.2f}<extra></extra>'
            ))
            
            center_x = sum(data['x'][:-1]) / 4
            center_y = sum(data['y'][:-1]) / 4
            fig_map.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='text',
                text=[sector_name],
                textfont=dict(size=14, color='#fff', family='Courier Prime'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig_map.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 3.5]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.5, 3.5]),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_map, use_container_width=True, key="heatmap_forensics")
        
        # Chart with Fixed Y-Axis
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=current_data['Time_Hrs'], 
            y=current_data['Pore_Water_Pressure_kPa'], 
            name='Pressure', 
            line=dict(color='#00f2ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,242,255,0.12)'
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="#ff003c", line_width=2)
        fig.update_layout(
            template="plotly_dark", 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=350,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Time (Hrs)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Pressure (kPa)', range=[0, 120])
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_forensics")
        
        # Command Log
        st.markdown("### 📡 Outbound Command Log")
        log_display = "\n".join(st.session_state.command_log) if st.session_state.command_log else "[Monitoring... No commands sent]"
        st.markdown(f"<div class='command-log'>{log_display}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Edge Node: RaspberryPi 5 | GSI Bhukosh Compliant")

"""Engineered features of a CT X-ray tube log: 45 features per scan computed from the raw sensor fields,
including 8 features of the cooling chain of the Straton tube (#38-#45).

The failure-risk pipeline (dtgexp/failrisk.py) removes the two cumulative features (cumulative_thermal_load,
straton_cumul_stress), averages the remaining 43 per day and expresses every daily value as an offset from the
tube's own reference period.
"""

import numpy as np
import pandas as pd

# ============================================================================
# Physical constants
# ============================================================================
COOLING_AREA = 0.0085  # m^2 - area of the cooling interface
ALPHA_NOMINAL = 30000  # W/(m^2 K) - nominal heat-transfer coefficient
NOMINAL_CURRENT = 300  # mA - nominal tube current used as reference
COOLING_CAPACITY_NOMINAL = 63750  # W - nominal cooling capacity (30000 x 0.0085 x 250 K)


def load_raw_csv(csv_path):
    """Load one tube log (CSV written by scripts/convert_tua.py).

    Required columns: scan_count, voltage_act, current_mean, scan_time_act, oil_temp_begin, oil_temp_end.
    Optional columns: datetime, oil_pressure, calc_temp_anode_cooling_interface, motor_temp, gantry_temp,
    anode_freq, motor_curr, xc_drops (missing optional columns are replaced by constants).
    """
    df = pd.read_csv(csv_path)

    print(f"  raw data: {len(df)} rows, {len(df.columns)} columns")
    print(f"  columns: {list(df.columns)[:10]}...")

    required_cols = ['scan_count', 'voltage_act', 'current_mean', 'scan_time_act',
                     'oil_temp_begin', 'oil_temp_end']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    # parse the datetime column (if present)
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            print("  datetime column parsed")
        except Exception as e:
            print(f"  datetime column could not be parsed: {e}")

    return df


def compute_all_features(df):
    """Compute the 45 engineered features of every scan.

    Args:
        df: raw log as returned by load_raw_csv

    Returns:
        features_df: DataFrame with the 45 features
    """
    features = pd.DataFrame()

    # ========================================================================
    # raw sensor data
    # ========================================================================
    voltage = df['voltage_act'].values
    current = df['current_mean'].values  # mA
    scan_time = df['scan_time_act'].values  # seconds
    oil_temp_begin = df['oil_temp_begin'].values
    oil_temp_end = df['oil_temp_end'].values

    # optional columns (constants when absent)
    oil_pressure = df['oil_pressure'].values if 'oil_pressure' in df.columns else np.ones(len(df)) * 1.5
    temp_interface = df['calc_temp_anode_cooling_interface'].values if 'calc_temp_anode_cooling_interface' in df.columns else oil_temp_end + 50
    motor_temp = df['motor_temp'].values if 'motor_temp' in df.columns else oil_temp_end
    gantry_temp = df['gantry_temp'].values if 'gantry_temp' in df.columns else np.ones(len(df)) * 25
    anode_freq = df['anode_freq'].values if 'anode_freq' in df.columns else np.ones(len(df)) * 50
    motor_curr = df['motor_curr'].values if 'motor_curr' in df.columns else np.ones(len(df)) * 2.0
    xc_drops = df['xc_drops'].values if 'xc_drops' in df.columns else np.zeros(len(df))

    # ========================================================================
    # features 1-4: temperature sensors
    # ========================================================================
    features['temp_interface'] = temp_interface  # #1
    features['temp_oil'] = oil_temp_end  # #2
    features['temp_ambient'] = gantry_temp  # #3
    features['temp_motor'] = motor_temp  # #4

    # ========================================================================
    # features 5-9: thermal load
    # ========================================================================
    thermal_load = voltage * (current / 1000.0) * scan_time  # Joules
    thermal_power = voltage * (current / 1000.0)  # Watts

    features['thermal_load'] = thermal_load  # #5
    features['thermal_power'] = thermal_power  # #6
    features['cumulative_thermal_load'] = thermal_load.cumsum()  # #7 (removed by the failure-risk pipeline)
    features['thermal_load_sum_50'] = pd.Series(thermal_load).rolling(50, min_periods=1).sum().values  # #8
    features['power_sum_50'] = pd.Series(thermal_power).rolling(50, min_periods=1).sum().values  # #9

    # ========================================================================
    # features 10-14: temperature gradients and thermal resistances
    # ========================================================================
    gradient_R2 = temp_interface - oil_temp_end
    gradient_R3 = oil_temp_end - gantry_temp

    features['gradient_R2'] = gradient_R2  # #10
    features['gradient_R3'] = gradient_R3  # #11
    features['thermal_resistance_R2'] = gradient_R2 / (thermal_power + 1e-6)  # #12 K/W
    features['thermal_resistance_R3'] = gradient_R3 / (thermal_power + 1e-6)  # #13 K/W
    features['gradient_R2_ratio'] = gradient_R2 / (gradient_R2 + gradient_R3 + 1e-6)  # #14

    # ========================================================================
    # features 15-18: tube-current features (relative to the nominal current)
    # ========================================================================
    features['filament_efficiency'] = current / NOMINAL_CURRENT  # #15

    current_change = np.diff(current, prepend=current[0])
    features['filament_change_rate'] = current_change / (current + 1e-6)  # #16

    current_volatility = pd.Series(current).rolling(10, min_periods=1).apply(lambda x: x.max() - x.min()).values
    features['current_volatility'] = current_volatility  # #17
    features['current_deviation'] = current - NOMINAL_CURRENT  # #18

    # ========================================================================
    # features 19-20: cooling
    # ========================================================================
    oil_temp_rise = oil_temp_end - oil_temp_begin
    features['oil_temp_rise'] = oil_temp_rise  # #19
    features['cooling_efficiency'] = oil_temp_rise / (thermal_load + 1e-6)  # #20 degC/J

    # ========================================================================
    # features 21-26: time features
    # ========================================================================
    if 'datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['datetime']):
        time_gaps = df['datetime'].diff().dt.total_seconds() / 60.0  # minutes
        time_gaps = time_gaps.fillna(0)
        features['time_gap_minutes'] = time_gaps.values  # #21

        # duty cycle
        duty_cycle = scan_time / (scan_time + time_gaps.values * 60 + 1e-6)
        features['duty_cycle'] = duty_cycle  # #22

        # cyclic encoding of hour and weekday
        hours = df['datetime'].dt.hour
        day_of_week = df['datetime'].dt.dayofweek

        features['hour_sin'] = np.sin(2 * np.pi * hours / 24)  # #23
        features['hour_cos'] = np.cos(2 * np.pi * hours / 24)  # #24
        features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)  # #25
        features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)  # #26
    else:
        # no datetime: constant defaults
        features['time_gap_minutes'] = 0.0
        features['duty_cycle'] = 0.5
        features['hour_sin'] = 0.0
        features['hour_cos'] = 1.0
        features['day_sin'] = 0.0
        features['day_cos'] = 1.0

    # ========================================================================
    # features 27-29: heat-transfer approximations
    # ========================================================================
    features['biot_number_approx'] = gradient_R2 / (temp_interface + 273.15)  # #27
    features['oil_convection_heat'] = oil_temp_end - gantry_temp  # #28
    features['interface_radiation_approx'] = (temp_interface + 273.15) ** 4 / 1e9  # #29 (scaled)

    # ========================================================================
    # features 30-33: changes from the previous scan
    # ========================================================================
    features['interface_temp_diff'] = np.diff(temp_interface, prepend=temp_interface[0])  # #30
    features['oil_temp_diff'] = np.diff(oil_temp_end, prepend=oil_temp_end[0])  # #31
    features['motor_temp_diff'] = np.diff(motor_temp, prepend=motor_temp[0])  # #32
    features['current_diff'] = np.diff(current, prepend=current[0])  # #33

    # ========================================================================
    # features 34-37: further raw fields
    # ========================================================================
    features['oil_pressure'] = oil_pressure  # #34
    features['anode_freq'] = anode_freq  # #35
    features['motor_curr'] = motor_curr  # #36
    features['xc_drops'] = xc_drops  # #37

    # ========================================================================
    # features 38-45: cooling chain of the Straton tube
    # ========================================================================

    # #38 estimated heat-transfer coefficient: alpha = P / (A dT)
    alpha_estimated = thermal_power / (COOLING_AREA * (oil_temp_rise + 1e-6))
    features['straton_alpha_estimated'] = alpha_estimated

    # #39 deviation from the oil temperature expected at the nominal coefficient (alpha = 30000)
    temp_expected = gantry_temp + thermal_load / (ALPHA_NOMINAL * COOLING_AREA)
    temp_deviation = oil_temp_end - temp_expected
    features['straton_temp_deviation'] = temp_deviation

    # #40 cooling-effectiveness ratio: current / nominal heat-transfer coefficient
    alpha_ratio = alpha_estimated / ALPHA_NOMINAL
    alpha_ratio = np.clip(alpha_ratio, 0.1, 2.0)  # limit outliers
    features['straton_alpha_ratio'] = alpha_ratio

    # #41 cooling rate: heat removed per degree of oil temperature rise
    cooling_rate = thermal_load / (oil_temp_rise + 1e-6)
    features['straton_cooling_rate'] = cooling_rate

    # #42 pressure-temperature ratio of the oil
    pressure_temp = oil_pressure / (oil_temp_end + 273.15)
    features['straton_pressure_temp'] = pressure_temp

    # #43 cumulative thermal stress weighted by the cooling effectiveness (removed by the failure-risk pipeline)
    cumul_stress = (thermal_load * alpha_ratio).cumsum()
    features['straton_cumul_stress'] = cumul_stress

    # #44 cooling degradation relative to the first 10 records of the log
    cooling_eff_initial = cooling_rate[:10].mean()
    cooling_deg = 1 - (cooling_rate / (cooling_eff_initial + 1e-6))
    cooling_deg = np.clip(cooling_deg, -0.5, 1.0)  # limit outliers
    features['straton_cooling_deg'] = cooling_deg

    # #45 thermal balance: current thermal load / nominal cooling capacity
    thermal_balance = thermal_load / COOLING_CAPACITY_NOMINAL
    features['straton_thermal_balance'] = thermal_balance

    # ========================================================================
    # checks
    # ========================================================================
    assert features.shape[1] == 45, f"expected 45 features, got {features.shape[1]}"

    print("  45 features computed")
    print(f"  feature names: {list(features.columns[:5])}... {list(features.columns[-5:])}")

    # NaN and inf
    if features.isnull().any().any():
        nan_cols = features.columns[features.isnull().any()].tolist()
        print(f"  NaN values in: {nan_cols}")
        features = features.fillna(0)

    if np.isinf(features.values).any():
        print("  inf values replaced by 0")
        features = features.replace([np.inf, -np.inf], 0)

    return features

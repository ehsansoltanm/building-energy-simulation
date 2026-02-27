import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
# Define temperature thresholds
HEATING_SETPOINT = 20.0
COOLING_SETPOINT = 25.0
NAT_VENT_MIN_OUTDOOR_TEMP = 15.0
NAT_VENT_MAX_OUTDOOR_TEMP = 24.0
NAT_VENT_MAX_INDOOR_TEMP = 26.0
MECH_VENT_MIN_OUTDOOR_TEMP = -5.0 # Example threshold for HR
MECH_VENT_MAX_OUTDOOR_TEMP = 30.0 # Example threshold for HR
HEAT_RECOVERY_EFFICIENCY = 0.7 # Example efficiency
SHADING_SOLAR_THRESHOLD = 300 # W/m^2
SHADING_INDOOR_TEMP_THRESHOLD = 24.0

# Define file paths (assuming they are in the same directory)
EPW_FILE = "/home/ubuntu/upload/Torino_IT-hour.epw"
SIM_FILE = "/home/ubuntu/upload/eplusout.csv"
OUTPUT_FILE = "/home/ubuntu/control_strategy_output.txt"

# --- Helper Functions ---
def load_data(epw_path, sim_path):
    """Loads and preprocesses weather and simulation data."""
    try:
        epw = pd.read_csv(epw_path, skiprows=8, header=None)
        epw.columns = ["Year", "Month", "Day", "Hour", "Minute", "DataSource", "DryBulb", "DewPoint", "RH", "AtmosPressure", 
                       "ExtGlobHorRad", "ExtDirNormRad", "ExtDifHorRad", "GlobalHorRad", "DirectNormRad", 
                       "DiffuseHorRad", "InfraSky", "WindDir", "WindSpd", "TotalSkyCover", "OpaqueSkyCover", 
                       "Visibility", "CeilingHeight", "PresWeatherObs", "PresWeatherCodes", "PrecipWater", "AerosolOptDepth",
                       "SnowDepth", "DaysSinceSnow", "Albedo", "LiquidPrecip", "RainRate", "RainDuration", "SnowRate", "SnowDuration"]
        weather_df = epw[["DryBulb", "GlobalHorRad"]].copy()
        weather_df.columns = ["outdoor_temp_c", "global_solar_rad_w_m2"]
    except FileNotFoundError:
        print(f"Error: EPW file not found at {epw_path}")
        return None

    try:
        sim_df = pd.read_csv(sim_path, low_memory=False)
        # Try to find the correct temperature and occupancy columns
        temp_col = next((c for c in sim_df.columns if "Zone Mean Air Temperature" in c and "BLOCCO1:ZONA3" in c), None)
        occ_col = next((c for c in sim_df.columns if "Zone People Occupant Count" in c and "BLOCCO1:ZONA3" in c), None)
        
        if not temp_col:
            print("Error: Zone Mean Air Temperature column not found for BLOCCO1:ZONA3")
            return None
        if not occ_col:
            print("Warning: Zone People Occupant Count column not found for BLOCCO1:ZONA3. Assuming occupied.")
            # Create a dummy occupancy column if not found
            sim_df["occupancy_status"] = 1 
            occ_col = "occupancy_status"
            
        sim_df_hourly = sim_df[[temp_col, occ_col]].copy()
        sim_df_hourly.columns = ["indoor_temp_c", "occupancy_status"]
        # Convert occupancy count to boolean status (assuming > 0 means occupied)
        sim_df_hourly["occupancy_status"] = sim_df_hourly["occupancy_status"] > 0
        
    except FileNotFoundError:
        print(f"Error: Simulation file not found at {sim_path}")
        return None
    except StopIteration:
         print("Error: Could not find required columns in simulation file.")
         return None

    # Combine
    weather_df_reset = weather_df.reset_index(drop=True)
    sim_df_hourly_reset = sim_df_hourly.reset_index(drop=True)
    min_length = min(len(weather_df_reset), len(sim_df_hourly_reset))
    full_df = pd.concat([weather_df_reset.iloc[:min_length], sim_df_hourly_reset.iloc[:min_length]], axis=1).dropna()
    return full_df

def determine_actions(indoor_temp, outdoor_temp, solar_rad, occupied):
    """Determines control actions based on current state."""
    actions = {
        "heating": False,
        "cooling": False,
        "natural_ventilation": False,
        "mechanical_ventilation": False, # Assume always on if occupied, adjust HR based on temp
        "heat_recovery": False,
        "shading": False
    }

    if occupied:
        # Heating/Cooling based on indoor temp
        if indoor_temp < HEATING_SETPOINT:
            actions["heating"] = True
        elif indoor_temp > COOLING_SETPOINT:
            actions["cooling"] = True

        # Ventilation Strategy
        can_nat_vent = (NAT_VENT_MIN_OUTDOOR_TEMP <= outdoor_temp <= NAT_VENT_MAX_OUTDOOR_TEMP) and \
                       (indoor_temp < NAT_VENT_MAX_INDOOR_TEMP)
        
        if can_nat_vent and not actions["heating"] and not actions["cooling"]:
             actions["natural_ventilation"] = True
             actions["mechanical_ventilation"] = False # Prefer natural if possible
        else:
            actions["mechanical_ventilation"] = True # Always run mech vent if occupied and nat vent not possible
            # Heat Recovery Logic for Mechanical Ventilation
            if MECH_VENT_MIN_OUTDOOR_TEMP <= outdoor_temp <= MECH_VENT_MAX_OUTDOOR_TEMP:
                 # Consider HR if there's a significant temp difference and it's beneficial
                 temp_diff = abs(indoor_temp - outdoor_temp)
                 if temp_diff > 5: # Only use HR if temp difference is significant (e.g., > 5C)
                     if (actions["heating"] and outdoor_temp < indoor_temp) or \
                        (actions["cooling"] and outdoor_temp > indoor_temp):
                         actions["heat_recovery"] = True
            
        # Shading Control
        if solar_rad > SHADING_SOLAR_THRESHOLD and indoor_temp > SHADING_INDOOR_TEMP_THRESHOLD:
            actions["shading"] = True
            
    else: # Unoccupied
        # Minimal heating/cooling (e.g., setback temps - not implemented here for simplicity)
        # No ventilation needed
        actions["mechanical_ventilation"] = False
        actions["natural_ventilation"] = False
        # Activate shading if high solar radiation to prevent overheating passively
        if solar_rad > SHADING_SOLAR_THRESHOLD * 1.5: # Higher threshold when unoccupied
             actions["shading"] = True
             
    return actions

# --- Main Execution ---
data = load_data(EPW_FILE, SIM_FILE)

if data is not None:
    # Redirect print to file
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, "w") as f:
        sys.stdout = f
        
        print("--- Control Strategy Simulation (First 100 hours) ---")
        print(f"{'OutdoorT':<10} {'IndoorT':<10} {'Solar':<8} {'Occupied':<10} -> {'Heat':<6} {'Cool':<6} {'NatVent':<8} {'MechVent':<10} {'HR':<6} {'Shade':<6}")
        print("-"*80)

        for index, row in data.head(100).iterrows():
            indoor = row["indoor_temp_c"]
            outdoor = row["outdoor_temp_c"]
            solar = row["global_solar_rad_w_m2"]
            occupied = row["occupancy_status"]
            
            actions = determine_actions(indoor, outdoor, solar, occupied)
            
            print(f"{outdoor:<10.1f} {indoor:<10.1f} {solar:<8.0f} {str(occupied):<10} -> "
                  f"{str(actions['heating']):<6} {str(actions['cooling']):<6} {str(actions['natural_ventilation']):<8} "
                  f"{str(actions['mechanical_ventilation']):<10} {str(actions['heat_recovery']):<6} {str(actions['shading']):<6}")

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Control strategy output saved to {OUTPUT_FILE}")
else:
    print("Failed to load data. Control strategy simulation aborted.")


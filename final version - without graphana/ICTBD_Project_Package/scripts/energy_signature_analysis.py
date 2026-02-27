import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
# Define temperature thresholds
HEATING_BALANCE_POINT = 15.0
COOLING_BALANCE_POINT = 20.0

# Define file paths
EPW_FILE = "/home/ubuntu/upload/Torino_IT-hour.epw"
SIM_FILE = "/home/ubuntu/upload/eplusout.csv"
OUTPUT_DIR = "/home/ubuntu/output_energy_signature"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        weather_df = epw[["DryBulb"]].copy()
        weather_df.columns = ["outdoor_temp_c"]
    except FileNotFoundError:
        print(f"Error: EPW file not found at {epw_path}")
        return None

    try:
        sim_df = pd.read_csv(sim_path, low_memory=False)
        # Find heating and cooling columns (adjust names if necessary based on your IDF outputs)
        heating_col = next((c for c in sim_df.columns if "DistrictHeating:Facility" in c), None)
        cooling_col = next((c for c in sim_df.columns if "DistrictCooling:Facility" in c), None)
        
        if not heating_col or not cooling_col:
            print("Error: Heating or Cooling column not found in simulation file.")
            # Let's try the exact names from the pareto analysis
            heating_col = "DistrictHeating:Facility"
            cooling_col = "DistrictCooling:Facility"
            if heating_col not in sim_df.columns or cooling_col not in sim_df.columns:
                 print("Error: Could not find required columns even with exact names.")
                 return None
            else:
                 print("Found columns using exact names.")
            
        sim_df_hourly = sim_df[[heating_col, cooling_col]].copy()
        sim_df_hourly.columns = ["heating_j", "cooling_j"]
        
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

# --- Main Execution ---
data = load_data(EPW_FILE, SIM_FILE)

if data is not None:
    print("Data loaded successfully. Generating energy signature plots...")
    
    # Heating Signature
    heating_data = data[data["heating_j"] > 0]
    plt.figure(figsize=(10, 6))
    plt.scatter(heating_data["outdoor_temp_c"], heating_data["heating_j"], alpha=0.5)
    plt.axvline(HEATING_BALANCE_POINT, color='r', linestyle='--', label=f"Balance Point ({HEATING_BALANCE_POINT}°C)")
    plt.title("Heating Energy Signature")
    plt.xlabel("Outdoor Temperature (°C)")
    plt.ylabel("Heating Energy (J)")
    plt.legend()
    plt.grid(True)
    heating_plot_path = os.path.join(OUTPUT_DIR, "energy_signature_heating.png")
    plt.savefig(heating_plot_path)
    plt.close()
    print(f"Heating signature plot saved to {heating_plot_path}")

    # Cooling Signature
    cooling_data = data[data["cooling_j"] > 0]
    plt.figure(figsize=(10, 6))
    plt.scatter(cooling_data["outdoor_temp_c"], cooling_data["cooling_j"], alpha=0.5)
    plt.axvline(COOLING_BALANCE_POINT, color='r', linestyle='--', label=f"Balance Point ({COOLING_BALANCE_POINT}°C)")
    plt.title("Cooling Energy Signature")
    plt.xlabel("Outdoor Temperature (°C)")
    plt.ylabel("Cooling Energy (J)")
    plt.legend()
    plt.grid(True)
    cooling_plot_path = os.path.join(OUTPUT_DIR, "energy_signature_cooling.png")
    plt.savefig(cooling_plot_path)
    plt.close()
    print(f"Cooling signature plot saved to {cooling_plot_path}")

else:
    print("Failed to load data. Energy signature analysis aborted.")


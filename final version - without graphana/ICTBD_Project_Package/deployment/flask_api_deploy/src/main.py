from flask import Flask, request, jsonify
import sys
import os

# Required for deployment environment
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    # Import the control logic function from the copied file
    from src.control_strategy import determine_actions as get_control_actions
except ImportError as e:
    print(f"Warning: Failed to import control logic: {e}. Using dummy control logic.")
    # Define a dummy function if import fails
    def get_control_actions(outdoor_temp_c, indoor_temp_c, global_solar_rad_w_m2, occupancy_status):
        # In a real scenario, log this error properly
        return {"error": "Control logic module not available or failed to load"}

app = Flask(__name__)

# --- Global State (Example - In a real app, use a database or proper state management) ---
current_sensor_data = {
    "indoor_temperature_c": 21.0,
    "outdoor_temperature_c": 18.0,
    "global_solar_rad_w_m2": 150.0,
    "occupancy_status": True
}
last_control_actions = {}

# --- API Endpoints ---

@app.route("/")
def index():
    return "ICTBD Lab REST API Example is running!"

@app.route("/sensors", methods=["POST"])
def receive_sensor_data():
    """Receives sensor data via POST request.
    Expected JSON payload: {
        "indoor_temperature_c": float,
        "outdoor_temperature_c": float,
        "global_solar_rad_w_m2": float,
        "occupancy_status": bool
    }
    """
    global current_sensor_data
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Basic validation (add more robust validation as needed)
    required_keys = ["indoor_temperature_c", "outdoor_temperature_c", "global_solar_rad_w_m2", "occupancy_status"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required keys in payload"}), 400

    current_sensor_data.update(data)
    print(f"Received sensor data: {current_sensor_data}")
    return jsonify({"message": "Sensor data received successfully"}), 200

@app.route("/status", methods=["GET"])
def get_status():
    """Returns the latest sensor data and control actions.
    """
    return jsonify({
        "current_sensor_data": current_sensor_data,
        "last_control_actions": last_control_actions
    }), 200

@app.route("/control", methods=["GET"])
def get_new_control_actions():
    """Calculates and returns new control actions based on current sensor data.
    """
    global last_control_actions
    try:
        # Load necessary data for control strategy if needed (modify control_strategy if it reads files)
        # For now, assuming get_control_actions only needs the provided arguments
        actions = get_control_actions(
            current_sensor_data["outdoor_temperature_c"],
            current_sensor_data["indoor_temperature_c"],
            current_sensor_data["global_solar_rad_w_m2"],
            current_sensor_data["occupancy_status"]
        )
        last_control_actions = actions
        print(f"Calculated control actions: {actions}")
        return jsonify(actions), 200
    except Exception as e:
        print(f"Error calculating control actions: {e}")
        return jsonify({"error": "Failed to calculate control actions", "details": str(e)}), 500

# --- Main Execution (for local testing, deployment uses WSGI server) ---
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False) # Debug should be False for production


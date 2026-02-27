# Technical Guide: EnergyPlus Gym and BMS Integration

This document provides a step-by-step technical guide for implementing the more complex parts of the project: integrating EnergyPlus with OpenAI Gym for stepped simulation and connecting the system components within a Building Management System (BMS) framework using openHAB, InfluxDB, and Grafana. These steps were identified as challenging to fully implement and test within this environment.

## Part 1: EnergyPlus Gym Integration for Stepped Simulation

EnergyPlus Gym allows you to wrap an EnergyPlus simulation within an OpenAI Gym environment. This enables reinforcement learning or other control algorithms to interact with the simulation step-by-step, receiving observations (like temperatures, energy use) and sending actions (like setpoints, actuator states).

**Prerequisites:**

*   Python environment (e.g., Conda or venv) with necessary libraries.
*   EnergyPlus installed and accessible from the command line.
*   `energyplus_env` library installed (`pip install energyplus_env`).

**Steps:**

1.  **Prepare IDF File:**
    *   Ensure your IDF file (`Res_flat1.idf` or the optimized version) is configured for external interface control. This typically involves adding `ExternalInterface` objects for actuators (things you want to control, e.g., heating/cooling setpoints, window shades) and sensors (variables you want to observe, e.g., zone temperatures, energy consumption).
    *   **Example IDF Snippets:**
        ```idf
        ExternalInterface,
          FunctionalMockupUnitExport; !- Name of External Interface

        ExternalInterface:FunctionalMockupUnitExport:From:Variable,
          Heating_Setpoint,          !- EnergyPlus Variable Name
          Schedule Value,            !- Type of Variable
          Heating Setpoint Schedule; !- Name of Schedule

        ExternalInterface:FunctionalMockupUnitExport:To:Variable,
          Zone_Air_Temperature,      !- EnergyPlus Variable Name
          Zone Air Temperature,      !- Type of Variable
          BLOCCO1:ZONA3;             !- Key Value
        ```
    *   You need to define schedules that will be overridden by the external interface for actuators and specify the exact variables/outputs for sensors.

2.  **Create `variables.cfg`:**
    *   This file maps the variables defined in the IDF's `ExternalInterface` objects to names used by the Gym environment. It tells EnergyPlus which variables to expose.
    *   **Example `variables.cfg`:**
        ```
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <fmuVariables>
          <variable name="Heating_Setpoint" causality="input" type="Real">
            <description>Heating Setpoint Schedule Value</description>
          </variable>
          <variable name="Zone_Air_Temperature" causality="output" type="Real">
            <description>Zone Air Temperature for BLOCCO1:ZONA3</description>
          </variable>
          <!-- Add other input/output variables here -->
        </fmuVariables>
        ```

3.  **Write Python Control Script:**
    *   Import `energyplus_env` and `gym`.
    *   Define the environment using `gym.make()`:
        ```python
        import gym
        import energyplus_env # Important to register the env

        env = gym.make(
            "EnergyPlus-v0",
            idf_path="/path/to/your/Res_flat1_optimized.idf",
            epw_path="/path/to/your/Torino_IT-hour.epw",
            variables_cfg_path="/path/to/your/variables.cfg",
            # Optional: Define simulation start/end month/day if needed
            # start_month=1,
            # start_day=1,
            # end_month=12,
            # end_day=31,
        )
        ```
    *   Implement the standard Gym interaction loop:
        ```python
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            # --- Your Control Logic Here ---
            # 1. Get current state from observation (e.g., temperature)
            current_temp = observation[0] # Assuming temp is the first observed variable

            # 2. Use your control strategy (control_strategy_v1.py logic)
            #    or prediction model (lstm_prediction_online_v1.ipynb logic)
            #    to determine the next action (e.g., heating setpoint).
            #    This might involve calling predict_next_step() and then the control logic.
            action = [21.0] # Example: Set heating setpoint to 21.0

            # 3. Step the environment
            observation, reward, done, info = env.step(action)

            # 4. Process results (e.g., log data, update total reward)
            total_reward += reward
            print(f"Observation: {observation}, Reward: {reward}, Done: {done}")

        env.close()
        print(f"Simulation finished. Total reward: {total_reward}")
        ```
    *   **Key Challenge:** Mapping the `observation` space (what you get from `env.step`) and `action` space (what you send to `env.step`) correctly based on your `variables.cfg`. The order matters.
    *   **Reward Function:** The default reward is often negative energy consumption. You might need to customize this by modifying the `energyplus_env` library or creating a custom Gym wrapper.

4.  **Run the Script:** Execute the Python script. EnergyPlus will run in the background, controlled by your script.

**Troubleshooting:**

*   **EnergyPlus Errors:** Check `eplusout.err` generated during the run.
*   **Variable Mismatches:** Ensure `variables.cfg` exactly matches the names and order expected by your Python script and the IDF.
*   **Environment Setup:** Complex dependencies might arise, especially with different EnergyPlus versions.

## Part 2: BMS Integration (openHAB, InfluxDB, Grafana)

This involves connecting the simulation/control loop (from Part 1) to a BMS platform for data storage, visualization, and potentially more complex automation rules.

**Conceptual Workflow:**

1.  **Data Flow:**
    *   **EnergyPlus Gym -> Python Script:** Provides sensor readings (observations).
    *   **Python Script -> Prediction Model:** Sends recent sensor data to `predict_next_step`.
    *   **Prediction Model -> Python Script:** Returns predicted energy/comfort range.
    *   **Python Script -> Control Strategy:** Sends current state and predictions to `determine_actions`.
    *   **Control Strategy -> Python Script:** Returns actuator commands (actions).
    *   **Python Script -> EnergyPlus Gym:** Sends actions to the simulation.
    *   **Python Script -> MQTT/REST:** Publishes sensor data, predictions, and actions to the BMS.
    *   **BMS (openHAB) -> MQTT/REST:** Subscribes to data topics/endpoints.
    *   **BMS (openHAB) -> InfluxDB:** Persists received data using the InfluxDB persistence service.
    *   **Grafana -> InfluxDB:** Queries data for visualization.

**Implementation Steps (High-Level):**

1.  **Set up BMS Components:**
    *   **InfluxDB:** Install and configure InfluxDB (version 1.x or 2.x). Create a database (e.g., `ictbd_lab`) and user credentials.
    *   **Grafana:** Install and configure Grafana. Add InfluxDB as a data source, connecting it to your database.
    *   **openHAB:** Install and configure openHAB. Install necessary bindings: MQTT binding, REST Docs, InfluxDB persistence.

2.  **Configure openHAB:**
    *   **MQTT Broker Connection:** Configure the MQTT binding to connect to your MQTT broker (e.g., `mqtt://test.mosquitto.org` or a local broker).
    *   **InfluxDB Persistence:** Configure the InfluxDB persistence service (`influxdb.cfg` or via UI) to connect to your InfluxDB instance and specify which items to persist (e.g., persist every change for relevant items).
    *   **Create Items:** Define openHAB items corresponding to the data you want to manage (e.g., `Number Zone_Temperature`, `Number Heating_Setpoint`, `String Control_Mode`).
    *   **Link Items to MQTT/REST:**
        *   **MQTT:** Create Generic MQTT Things and configure channels to subscribe to topics published by your Python script (e.g., `ictbd/lab/group4/sensors/temperature`) and link them to your items.
        *   **REST:** openHAB's REST API can be used, but MQTT is often simpler for this type of data streaming.

3.  **Modify Python Control Script (from Part 1):**
    *   **Add Communication:** Integrate the MQTT client (`mqtt_example_v1.py` logic) or REST calls within the Gym loop.
    *   **Publish Data:** After each `env.step()`, publish the latest `observation`, `action`, and potentially the `prediction` results (from `predict_next_step`) to relevant MQTT topics.
        ```python
        # Inside the while loop after env.step()
        mqtt_client.publish(f"ictbd/lab/group4/sensors/temperature", payload=str(observation[0]))
        mqtt_client.publish(f"ictbd/lab/group4/actions/heating_setpoint", payload=str(action[0]))
        # Publish prediction results if available
        if 'prediction' in locals():
             mqtt_client.publish(f"ictbd/lab/group4/predictions/median", payload=str(prediction['median']))
             mqtt_client.publish(f"ictbd/lab/group4/predictions/lower", payload=str(prediction['lower']))
             mqtt_client.publish(f"ictbd/lab/group4/predictions/upper", payload=str(prediction['upper']))
        ```
    *   **Optional: Subscribe to Control Topics:** If you want openHAB rules to override the Python script's control logic, subscribe to control topics (e.g., `ictbd/lab/group4/control/setpoint_override`) and adjust the `action` accordingly.

4.  **Create Grafana Dashboards:**
    *   Log in to Grafana.
    *   Create a new dashboard.
    *   Add panels (e.g., Graph, Singlestat).
    *   Configure panels to query data from your InfluxDB data source. Use Flux or InfluxQL queries to select the data linked to your openHAB items (e.g., `SELECT value FROM "Zone_Temperature" WHERE time > now() - 1h`).
    *   Visualize temperature trends, energy predictions, control actions, etc.

**Key Challenges:**

*   **Component Setup:** Installing and configuring openHAB, InfluxDB, and Grafana correctly can be complex, especially regarding networking and service management.
*   **Data Synchronization:** Ensuring data flows correctly and timely between all components.
*   **Debugging:** Identifying issues across multiple systems (Python, EnergyPlus, MQTT, openHAB, InfluxDB, Grafana) requires careful logging and monitoring.
*   **Resource Intensity:** Running EnergyPlus simulation, Python scripts, and the BMS components simultaneously can be resource-intensive.

This guide provides the foundational steps. Each step involves detailed configuration specific to your setup and the exact variables/controls you implement. Refer to the official documentation for EnergyPlus, `energyplus_env`, openHAB, InfluxDB, and Grafana for more specific instructions.

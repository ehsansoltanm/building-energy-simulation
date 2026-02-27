import paho.mqtt.client as mqtt
import time
import json
import random

# --- Configuration ---
MQTT_BROKER = "mqtt.eclipseprojects.io" # Public test broker
MQTT_PORT = 1883
SENSOR_TOPIC = "ictbd/lab/group4/sensors" # Replace X with your group number
CONTROL_TOPIC = "ictbd/lab/group4/control" # Replace X with your group number
CLIENT_ID = f"ictbd_client_{random.randint(0, 1000)}"

# --- Callback Functions ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
        # Subscribe to control topic upon connection
        client.subscribe(CONTROL_TOPIC)
        print(f"Subscribed to topic: {CONTROL_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message on topic 	{msg.topic}	: {msg.payload.decode()}")
    # Add logic here to handle received control commands
    try:
        command = json.loads(msg.payload.decode())
        print(f"Parsed command: {command}")
        # Example: if command.get("heating") == "on": ...
    except json.JSONDecodeError:
        print("Error decoding JSON command")

# --- Main Script ---
def run_mqtt_client():
    client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        return

    # Start the network loop in a separate thread
    client.loop_start()

    # Simulate publishing sensor data periodically
    try:
        while True:
            # Simulate sensor readings
            indoor_temp = round(random.uniform(18.0, 25.0), 1)
            outdoor_temp = round(random.uniform(5.0, 30.0), 1)
            sensor_data = {
                "indoor_temperature_c": indoor_temp,
                "outdoor_temperature_c": outdoor_temp,
                "timestamp": time.time()
            }
            payload = json.dumps(sensor_data)
            result = client.publish(SENSOR_TOPIC, payload)
            status = result.rc
            if status == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published to {SENSOR_TOPIC}: {payload}")
            else:
                print(f"Failed to send message to topic {SENSOR_TOPIC}, status: {status}")

            time.sleep(10) # Publish every 10 seconds

    except KeyboardInterrupt:
        print("Disconnecting from MQTT broker...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    run_mqtt_client()


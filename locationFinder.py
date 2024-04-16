import paho.mqtt.client as mqtt
from numpy import full, average
import json
from location_algorithm import locate
import mqtt_credentials as credentials

beaconIDtoNumber = {
    '8cda': 0,
    '60d8': 1,
    '8860': 2,
    '8cd0': 3,
    'a8d9': 4
}

defaultRSSIValue = -120

# This method is called when the connection is (re)stablished to the server
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected")
    client.subscribe(credentials.subscribe_topic)

def format_measurements(measurements):
    measurement_averages = full((1,5),-120)
    for measurement in measurements:
        if measurement["id"] not in beaconIDtoNumber:
            print(f'{measurement["id"]} not found')
            continue
        index = beaconIDtoNumber[measurement["id"]]
        measurement_averages[0][index] = average(measurement["rssi"])
    return measurement_averages

# This method is called when a message is published to the topic
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)["data"]
    rssiValues = format_measurements(payload)
    print(locate(rssiValues))

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.username_pw_set(credentials.username, credentials.password)
mqttc.connect(credentials.server, credentials.port, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
mqttc.loop_forever()
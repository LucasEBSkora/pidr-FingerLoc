import paho.mqtt.client as mqtt
import pandas as pd
import json
from os.path import exists
import sys

import mqtt_credentials as credentials


beaconIDtoNumber = {
    '8cda': 0,
    '60d8': 1,
    '8860': 2,
    '8cd0': 3,
    'a8d9': 4
}

defaultRSSIValue = -120
numberOfSamples = 20


if len(sys.argv) < 2:
    print("missing path to measurement file!\nusage: python fingerprint.py path/to/measurement/file.csv")
fingerprint_file_path = sys.argv[1]


# This method is called when the connection is (re)stablished to the server
# it connects to the topic where the rssi data is published
def on_connect(client, userdata, flags, reason_code, properties):
    print("connected")
    client.subscribe(credentials.subscribe_topic)

# This method is called when a message is published to the topic
def on_message(client, userdata, msg):
    fingerprints = pd.DataFrame(columns=["ID", "x", "y", "rssi1", "rssi2", "rssi3", "rssi4", "rssi5"])
    measurements = json.loads(msg.payload)["data"]
    measurement_id = measurements[0]["measure_id"]
    print(str(measurements))
    x, y = get_coordinates()
    if x is None:
        return
    measurements = order_measurements(measurements)
    print(str(measurements))
    for i in range(0, 20):
        row = pd.Series({"ID": measurement_id, "x": x, "y": y, "rssi1": -120, "rssi2": -120, "rssi3": -120, "rssi4": -120, "rssi5": -120,})
        for j in range(0,5):
            row[f"rssi{j+1}"] = measurements[j]["rssi"][i]
        fingerprints.loc[i] = row
    print(fingerprints)
    if exists(fingerprint_file_path):
        fingerprints.to_csv(fingerprint_file_path, mode='a', header=False)
    else:
        fingerprints.to_csv(fingerprint_file_path, mode='w')

def get_coordinates():
    val = input("please write the coordinates of the point in the following format: x,y\nor write 'ignore' to ignore this measurement\n")
    while True:
        if val.find("ignore") != -1:
            return (None, None)
        params = val.split(",")
        if  len(params) < 2:
            val = input("not enough parameters! try again\n")
            continue
        x = float(params[0])
        y = float(params[1])
        n = 1
        return (x, y)

# makes sure the measurements received are ordered the same way as we number the beacons.
# if any of the beacons are missing from the current measurement, assumes all values are -120
def order_measurements(measurements):
    beacons = list([[defaultRSSIValue]*numberOfSamples]*5)
    for measurement in measurements:
        if measurement["id"] not in beaconIDtoNumber:
            print(f'{measurement["id"]} not found')
            continue
        index = beaconIDtoNumber[measurement["id"]]
        beacons[index] = measurement
    return beacons

# creates the MQTT objects to connect to the topics
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.username_pw_set(credentials.username, credentials.password)
mqttc.connect(credentials.server, credentials.port, 60)

mqttc.loop_forever()
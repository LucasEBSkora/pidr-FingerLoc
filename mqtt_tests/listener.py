import paho.mqtt.client as mqtt

import src.mqtt_credentials as credentials

# This method is called when the connection is (re)stablished to the server
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    client.subscribe(credentials.subscribe_topic)

# This method is called when a message is published to the topic
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

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
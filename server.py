'''
from opcua import Server,ua
from datetime import datetime
import time
import matplotlib.pyplot as plt
import csv

# Remove global variables for the OPC UA server and variables (temporarily)
# server = None
# conv1_active_power = None
# conv1_kq_set = None
# conv1_reactive_power = None
# conv1_voltage_meas = None
# conv2_active_power = None
# conv2_kq_set = None
# conv2_reactive_power = None
# conv2_voltage_meas = None

class myServer:
    def __init__(self):
        self.conv1_active_power = None
        self.conv1_kq_set = None
        self.conv1_reactive_power = None
        self.conv1_voltage_meas = None
        self.conv2_active_power = None
        self.conv2_kq_set = None
        self.conv2_reactive_power = None
        self.conv2_voltage_meas = None
        self.conv1_kq_set_list = []

        # Setup the OPC UA server
        self.server = Server()
        
        # Set endpoint for the server
        
    def start_server(self):
        #global server, conv1_active_power, conv1_kq_set, conv1_reactive_power, conv1_voltage_meas
        #global conv2_active_power, conv2_kq_set, conv2_reactive_power, conv2_voltage_meas
        self.server.set_endpoint("opc.tcp://localhost:4846/freeopcua/server/")
        print("Server endpoint: {}".format(self.server.endpoint))
        
        # Setup the server namespace
        uri = "http://example.org"
        self.idx = self.server.register_namespace(uri)
        
        # Get the Objects node, this is where we should add our nodes
        self.objects = self.server.get_objects_node()
        
        # Create a new object in the server
        self.my_object = self.objects.add_object(self.idx, "MyObject")
        
        
        
        # Create variables in the new object
        self.conv1_active_power = self.my_object.add_variable(self.idx, "CONV_1_ACTIVE_POWER", 0.00)
        self.conv1_kq_set = self.my_object.add_variable(self.idx, "CONV_1_KQ_SET", 0.00)
        self.conv1_reactive_power = self.my_object.add_variable(self.idx, "CONV_1_REACTIVE_POWER", 0.00)
        self.conv1_voltage_meas = self.my_object.add_variable(self.idx, "CONV_1_VOLTAGE_MEAS", 0.00)

        self.conv2_active_power = self.my_object.add_variable(self.idx, "CONV_2_ACTIVE_POWER", 0.00)
        self.conv2_kq_set = self.my_object.add_variable(self.idx, "CONV_2_KQ_SET", 0.00)
        self.conv2_reactive_power = self.my_object.add_variable(self.idx, "CONV_2_REACTIVE_POWER", 0.00)
        self.conv2_voltage_meas = self.my_object.add_variable(self.idx, "CONV_2_VOLTAGE_MEAS", 0.00)
        
        # Set writable properties
        self.conv1_active_power.set_writable()
        self.conv1_kq_set.set_writable()  # Readable and writable
        self.conv1_reactive_power.set_writable()
        self.conv1_voltage_meas.set_writable()

        self.conv2_active_power.set_writable()
        self.conv2_kq_set.set_writable()  # Readable and writable
        self.conv2_reactive_power.set_writable()
        self.conv2_voltage_meas.set_writable()

        # Dynamically add CONV_1_KQ_SET_1 to CONV_1_KQ_SET_100 keys
        for i in range(1,10):
            kq_key = self.my_object.add_variable(self.idx, f"CONV_1_KQ_SET_{i}", 0.00)
            kq_key.set_writable()  # Make each variable writable
            self.conv1_kq_set_list.append(kq_key)
        
        
        # Start the server
        self.server.start()
        print("Server started at {}".format(self.server.endpoint))


    # Functions to get the current values of the variables
    def get_conv1_active_power(self):
        return self.conv1_active_power.get_value() if self.conv1_active_power else None

    def get_conv1_kq_set(self):
        return self.conv1_kq_set.get_value() if self.conv1_kq_set else None

    def get_conv1_reactive_power(self):
        return self.conv1_reactive_power.get_value() if self.conv1_reactive_power else None

    def get_conv1_voltage_meas(self):
        return self.conv1_voltage_meas.get_value() if self.conv1_voltage_meas else None

    def get_conv2_active_power(self):
        return self.conv2_active_power.get_value() if self.conv2_active_power else None

    def get_conv2_kq_set(self):
        return self.conv2_kq_set.get_value() if self.conv2_kq_set else None

    def get_conv2_reactive_power(self):
        return self.conv2_reactive_power.get_value() if self.conv2_reactive_power else None

    def get_conv2_voltage_meas(self):
        return self.conv2_voltage_meas.get_value() if self.conv2_voltage_meas else None
    
    def set_conv1_kq_set(self,value):
        return self.conv1_kq_set.set_value(value)
     # Function to set specific CONV_1_KQ_SET_X values
    def set_conv1_kq_set_by_index(self, index, value):
        if 1 <= index <= 50:
            self.conv1_kq_set_list[index - 1].set_value(value)
        else:
            print("Index out of range. Please provide an index between 1 and 100")
'''

import logging
from asyncua import Server, ua
import asyncio


class myAsyncServer:
    def __init__(self):
        self.server = Server()
        self.conv1_active_power = None
        self.conv1_kq_set = None
        self.conv1_reactive_power = None
        self.conv1_voltage_meas = None
        self.conv1_kq_set_1 = None  # Single tag CONV_1_KQ_SET_1

    async def start_server(self):
        # Set the server's endpoint
        await self.server.init()
        self.server.set_endpoint("opc.tcp://localhost:4846/freeopcua/server/")
        print(f"Server endpoint: {self.server.endpoint}")

        # Setup the server namespace
        uri = "http://example.org"
        idx = await self.server.register_namespace(uri)

        # Get the Objects node, this is where we should add our nodes
        objects_node = self.server.nodes.objects

        # Create a new object in the server
        my_object = await objects_node.add_object(idx, "MyObject")

        # Create variables in the new object
        self.conv1_active_power = await my_object.add_variable(idx, "CONV_1_ACTIVE_POWER", 0.0)
        self.conv1_kq_set = await my_object.add_variable(idx, "CONV_1_KQ_SET", 0.0)
        self.simulation_time = await my_object.add_variable(idx, "SIMULATION_TIME", 0.0)
        self.conv1_voltage_meas = await my_object.add_variable(idx, "CONV_1_VOLTAGE_MEAS", 0.0)
        self.conv1_kq_set_1 = await my_object.add_variable(idx, "CONV_1_KQ_SET_1", 0.0)

        # Set writable properties
        await self.conv1_active_power.set_writable()
        await self.conv1_kq_set.set_writable()
        await self.simulation_time.set_writable()
        await self.conv1_voltage_meas.set_writable()
        await self.conv1_kq_set_1.set_writable()  # Make this key writable

        # Start the OPC UA server
        await self.server.start()
        print("Server started at {}".format(self.server.endpoint))

    async def stop_server(self):
        # Stop the server when needed
        await self.server.stop()
        print("Server stopped")

    # Functions to get the current values of the variables
    async def get_conv1_active_power(self):
        return await self.conv1_active_power.read_value()

    async def get_conv1_kq_set(self):
        return await self.conv1_kq_set.read_value()

    async def get_simulation_time(self):
        return await self.simulation_time.read_value()

    async def get_conv1_voltage_meas(self):
        return await self.conv1_voltage_meas.read_value()

    # Function to set the value of CONV_1_KQ_SET_1
    async def set_conv1_kq_set_1(self, value):
        await self.conv1_kq_set_1.write_value(value)


# Main function to start and keep the server running
async def main():
    server_instance = myAsyncServer()
    await server_instance.start_server()

    try:
        # Keep the server running indefinitely until interrupted
        await asyncio.Event().wait()  # This waits forever until a signal is received
    except KeyboardInterrupt:
        print("Server interrupted. Shutting down...")
        await server_instance.stop_server()  # Stop the server gracefully


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())





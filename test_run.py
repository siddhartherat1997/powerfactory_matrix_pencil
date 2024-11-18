
import os
import pandas as pd
import asyncio
import time
import numpy as np
from scipy.linalg import hankel, svd
from math import ceil
from server import myAsyncServer  # Importing the updated async server
from pencil import Pencil  # Importing the Pencil function from pencil.py

# Initializing the Empty DataFrame
df = pd.DataFrame(columns=['Simulation_Time', 'Active_Power_(kW)'])

# Function to Append Data
def append_data(timestamp, active_power):
    global df

    # Check if timestamp already exists in the DataFrame and if active power is not zero
    if timestamp not in df['Simulation_Time'].values and active_power != 0:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Simulation_Time': [timestamp], 'Active_Power_(kW)': [active_power]})
        if not new_row.isna().all().all():
            # Concatenate the new row with the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Data appended: Simulation Time: {timestamp}, Active Power: {active_power}")
    else:
        print(f"Duplicate or zero power value detected at Simulation Time: {timestamp}")

# Function to Reset Buffer Data
def reset_buffer_data():
    global df
    if not df.empty:
        print(f'Buffer Reset. Total Collected: {len(df)} rows')
        df = df.iloc[0:0]  # Reset DataFrame after processing

# Main async function
async def main():
    print("Starting the server...")
    server = myAsyncServer()
    await server.start_server()

    previous_kq = None
    previous_active_power = None
    last_power_check_time = None  # Use simulation time for the power check interval
    kq_changed = False
    collecting_data = False
    collected_count = 0  # Count of collected values
    damping_values = []  # Initialize damping values list for plotting
    frequency_values = []
    initial_zeta = 40  # Start with an initial damping percentage higher than 7.5
    kq_set = False  # Flag to ensure kq is set once at -10 seconds
    recording_started = False  # Flag to start recording at -15 seconds
    last_kq_update_time = None  # Track when kq was last updated
    power_violation_detected = False  # Flag for power condition violation

    delta_kq = 0.005  # Step size for kq adjustment
    power_threshold = 0.5  # Threshold for change in active power

    while True:
        conv1_active_power = await server.get_conv1_active_power()
        simulation_time = await server.get_simulation_time()
        conv_1_voltage = await server.get_conv1_voltage_meas()
        current_kq = await server.get_conv1_kq_set()

        rounded_simulation_time = round(simulation_time, 2)
        rounded_active_power = round(conv1_active_power, 5)
        rounded_current_kq = round(current_kq, 3)

        # Start recording at t = -15 seconds
        if rounded_simulation_time == -15 and not recording_started:
            print(f"Starting recording at {rounded_simulation_time} seconds.")
            recording_started = True

        # If recording has started, continue collecting data
        if recording_started and rounded_simulation_time >= -15:
            # Set kq to 0.025 when simulation time reaches exactly -10 seconds
            if rounded_simulation_time == -10 and not kq_set:
                await server.set_conv1_kq_set_1(0.025)
                print(f"kq set to 0.025 at {rounded_simulation_time} seconds.")
                kq_set = True  # Set the flag to prevent resetting the value

            # Detect if kq has changed and set the flag
            if previous_kq is not None and previous_kq != rounded_current_kq:
                print(f"kq value changed from {previous_kq} to {rounded_current_kq}")
                kq_changed = True
                collected_count = 0  # Reset the collection count
                collecting_data = True  # Start collecting data
                last_kq_update_time = time.time()  # Record the time of the change

            # **Check if active power differs from the previous value by more than 0.5 every 2 simulation seconds**
            if previous_active_power is not None and (last_power_check_time is None or rounded_simulation_time - last_power_check_time >= 2):
                if abs(rounded_active_power - previous_active_power) > power_threshold:
                    print(f"Significant change in active power detected: {rounded_active_power}")
                    reset_buffer_data()
                    print(f"Buffer collection restarted due to active power change.")
                    collecting_data = True  # Start collecting data
                    power_violation_detected = True  # Mark that violation was detected
                last_power_check_time = rounded_simulation_time  # Update the last power check time

            previous_active_power = rounded_active_power  # Update previous active power

            # Collect 200 unique values after kq has changed or active power changes significantly
            if collecting_data and len(df) < 200:
                append_data(rounded_simulation_time, rounded_active_power)
                collected_count += 1

                if len(df) >= 200:
                    print("200 unique values collected. Running Pencil algorithm...")

                    # Pass the collected data to Pencil function and get the result
                    active_power_data = df['Active_Power_(kW)'].values
                    pencil_result = Pencil(active_power_data, Ts=0.01)
                    print(pencil_result)

                    # Logic for adjusting kq based on the output of the Pencil algorithm
                    if pencil_result.empty:
                        # If Pencil returns an empty DataFrame, increase kq by delta_kq
                        new_kq = current_kq + delta_kq
                        print(f"Pencil returned empty. Increasing kq by {delta_kq} to {new_kq}.")
                        await server.set_conv1_kq_set_1(new_kq)
                    else:
                        max_magnitude_row = pencil_result.iloc[0]
                        current_zeta = abs(max_magnitude_row['zeta (%)']) * 100
                        current_frequency = abs(max_magnitude_row['Frequency (Hz)'])
                        damping_values.append(current_zeta)  # Store damping values
                        frequency_values.append(current_frequency)
                        new_kq = current_kq  # Start with the current kq

                        if current_zeta > 8.05:
                            voltage_deviation = abs(conv_1_voltage - 19.82)
                            if voltage_deviation < 0.5:
                                new_kq += delta_kq  # Increase kq
                            elif voltage_deviation > 0.7:
                                new_kq -= delta_kq  # Decrease kq
                        elif current_zeta < 7.9:
                            new_kq -= delta_kq  # Decrease kq

                        print(f"New kq calculated: {new_kq}")
                        await server.set_conv1_kq_set_1(new_kq)
                        print(f"New kq sent to PowerFactory: {new_kq}")
                        last_kq_update_time = time.time()  # Reset last update time

                    # Reset the buffer after adjusting kq
                    reset_buffer_data()
                    collecting_data = False  # Stop collecting data until kq changes again
                    power_violation_detected = False  # Reset the violation flag

            # Check if kq hasn't changed within the last 10 seconds and increase kq
            if last_kq_update_time and (time.time() - last_kq_update_time >= 10):
                new_kq = current_kq + delta_kq
                print(f"No change in kq for 10 seconds, increasing kq to: {new_kq}")
                await server.set_conv1_kq_set_1(new_kq)
                print(f"New kq sent to PowerFactory: {new_kq}")
                last_kq_update_time = time.time()  # Update last kq update time

            previous_kq = rounded_current_kq  # Update previous_kq

        await asyncio.sleep(0.001)

# Starting the async loop when the script is run as a standalone program
if __name__ == "__main__":
    asyncio.run(main())







from utils import PowerFactoryUtils
import os

def main():
    # Define the path to the PowerFactory installation
    pf_path = r"F:\Digsilent\Python\3.11"
    
    # Define the project name
    project_path = r"MV_oscillations"
    
    # Initialize the PowerFactory utilities with the given paths
    pf_utils = PowerFactoryUtils(pf_path, project_path)

    # Create and set attributes for tags
    base_path = r"Network Model\Network Data\Grid\PCC-converter-1\Cub_1"
    pf_utils.create_and_set_tags(base_path, num_tags=3,create_tags_flag=False) # Select True to Create the OPC Tags Else False
    '''
    # Define the folder to save CSV files and the initial file counter
    location = r"F:\Thesis\Thesis Final\data_pipleline\buffer_readings"  # Update this path as needed
    file_counter = 1

    # Define time intervals for exporting simulation results
    from_interval = -15
    to_interval = -10

    # Call the save_result_during_simulation_to_csv function before running the simulation
    csv_file_path = pf_utils.save_result_during_simulation_to_csv(
        location=location,
        from_interval=from_interval,
        to_interval=to_interval,
        file_counter=file_counter
    )

    # Increment the file counter for the next result save
    file_counter += 1
    '''
    # Initialize the OPC simulation for time domain analysis
    pf_utils.initialize_opc_simulation({"iopt_type": 1})
    
    # Run the simulation for 80 seconds
    pf_utils.run_simulation(tstop=500)

if __name__ == "__main__":
    main()

import os
import sys
import powfacpy

class PowerFactoryUtils:

    def __init__(self, pf_path, project_path):
        sys.path.append(pf_path)
        import powerfactory
        self.app = powerfactory.GetApplication()
        self.app.Show()
        self.app.ActivateProject(project_path)
        self.pfp = powfacpy.PFActiveProject(self.app)
        self.pfpi = powfacpy.PFPlotInterface(self.app)
        self.pfds = powfacpy.PFDynSimInterface(self.app)

    def set_conv_attributes(self, gen_path, droop_value):
        """Set converter attributes."""
        self.pfp.set_attr(gen_path, {"mq": droop_value})
        
    def initialize_simulation(self, options=None):
        # Initialize the dynamic simulation with given options
        if options is None:
            options = {"iopt_sim": "rms"}
        self.pfds.initialize_sim(options)

    def initialize_opc_simulation(self, options=None):
        # Initialize the dynamic simulation with given options
        if options is None:
            options = {"iopt_type": "1"}
        self.pfds.initialize_opc(options)

    def create_and_set_tags(self, base_path, num_tags, create_tags_flag=False):
        """
        Creates and sets attributes for multiple StaExtdatmea objects with dynamic tags.
        
        Args:
        - base_path: The base path where the StaExtdatmea objects will be created.
        - num_tags: The number of tags to create (e.g. Conv_1_KQ_Set_1, Conv_1_KQ_Set_2).
        - create_tags_flag: Boolean flag to determine if the tags should be created.
        """
        if not create_tags_flag:
            print("Tag creation flag is set to False. No tags will be created.")
            return

        for i in range(1, num_tags + 1):
            # Generate tag names
            tag_name = f"Conv_1_KQ_Set_{i}"
            upper_tag_name = tag_name.upper()

            # Create a new object with the generated tag name
            new_obj_path = f"{base_path}\\{tag_name}.StaExtdatmea"
            new_obj = self.pfp.create_by_path(new_obj_path)

            # Fetch the Droop Control System object (make sure it exists)
            droop_control = self.pfp.get_obj(r"Network Model\Network Data\Grid\Droop Control System\Droop Control")[0]

            # Set attributes for the new object
            self.pfp.set_attr(new_obj, 
                             {"loc_name": tag_name,
                             "i_dat": 3, "outserv": 0,
                             "pCtrl": droop_control,
                             "iStatus": 1610612736,
                             "varName": "mq",
                             "deadband": 0.001,
                             "for_name": upper_tag_name,
                             "sTagID": upper_tag_name})

            print(f"Created and set attributes for: {upper_tag_name}")

        

    def run_simulation(self, tstop):
        # Run the dynamic simulation for a given stopping time
        self.pfds.run_sim({"tstop": tstop})

    def initialize_and_run_simulation(self, options=None, tstop=0.5):
        # Initialize and run the dynamic simulation
        self.initialize_opc_simulation(options)
        self.run_simulation(tstop)
    
    def save_result_during_simulation_to_csv(
        self, location: str, from_interval: float, to_interval: float, file_counter: int) -> str:
        """
        Save simulation result during a specific time interval to a CSV file.
        The file name increments automatically based on the given file_counter.

        Args:
            location (str): The directory location where the file will be saved.
            from_interval (float): The start time of the simulation interval.
            to_interval (float): The end time of the simulation interval.
            file_counter (int): The counter for the file naming (e.g., 1, 2, 3 for active_power_1, active_power_2, etc.).

        Returns:
            str: The file path of the saved CSV.
        """
        # Ensure the directory exists
        if not os.path.exists(location):
            os.makedirs(location)

        # Get the ComRes object from the study case (you can use self.app to interact with PowerFactory)
        comres: ComRes = self.app.GetFromStudyCase("ComRes")

        if not comres:
            raise Exception("ComRes object could not be found in the study case.")

        # Configure ComRes settings
        comres.iopt_csel = 1  # Export all variables
        comres.cfrom = from_interval  # Set the interval "from"
        comres.cto = to_interval  # Set the interval "to"

        # Define the file name and location for saving the CSV, using file_counter
        csv_file_name = f"active_power_{file_counter}.csv"
        csv_file_path = os.path.join(location, csv_file_name)
        comres.f_name = csv_file_path

        # Execute the ComRes command to export the data
        try:
            comres.Execute()  # Execute the export command
            print(f"Results successfully exported to {csv_file_path}")
            return csv_file_path  # Return the file path of the saved CSV

        except Exception as e:
            print(f"An error occurred while saving the CSV: {str(e)}")
            return None
    
    

    

    


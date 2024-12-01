# Powerfactory Automated Simulation Pipeline Using Matrix Pencil

**Authors:** Erat Siddharth Mannadiar, Prashant Pant, M.Sc, PhD Candidate  
**Chair:** Chair of Renewable and Sustainable Energy Systems  
**Affiliation:** Centre for Combined Smart Energy Systems (CoSES)  
**Role:** Research Coordinator - TUM SEED Centre  
**Institution:** Technical University of Munich


# Dependencies

This project currently runs on the following versions of Python and libraries:

- **Python**: 3.11.9
- **NumPy**: 1.26.4
- **Pandas**: 2.2.2
- **PyTorch**: 2.2.0+cu121
- **Matplotlib**: 3.9.2
- **Powpfacpy

# How to Run The Simulation

- Pip Install powfacpy
- Find the `dyn_sim_interface.py` file after installing powfacpy.
- Add this additional code in in the above mentioned python file:

```python
def initialize_opc(self, param=None):
    """
    Initialize time domain simulation.
    Parameters for 'ComInc' command object can be specified in 'param' dictionary.
    """
    comopc = self.app.GetFromStudyCase("ComOpc")
    if param is not None:
        self.set_attr(comopc, param)
    comopc.Execute()

```
- Load the pfd file(exported Powerfactory Simulation file) in powerfactory before using the code, make sure to change the powerfactory pathn in the `main.py` file.
- Pull/Download the respository.
- Run the `runner.py` file to start the simulation.
    

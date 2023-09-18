import os
import csv
from sklearn.model_selection import ParameterGrid
from NetEvolution import NetworkEvolution
from typing import List, Dict, Union, Tuple, Optional, Any

class ABMSweeps:

    """Class for generating parameter sweeps.
       This class allows you to specify a set of parameters for the ABM and generate
       a grid of combinations of these parameters for performing parameter sweeps by 
       calling the NetworkEvolution() class.
    
       Attributes:
        N (List[int]): List of values for the 'N' parameter.
        alpha (List[float]): List of values for the 'alpha' parameter.
        tau (List[int]): List of values for the 'tau' parameter.
        outdeg_l (List[float]): List of values for the 'outdeg_l' parameter.
        outdeg_h (List[float]): List of values for the 'outdeg_h' parameter.
        beta_attr_high (List[float]): List of values for the 'beta_attr_high' parameter.
        beta_exploit_low (List[float]): List of values for the 'beta_exploit_low' parameter.
        beta_explor_low (List[float]): List of values for the 'beta_explor_low' parameter.
        epsilon_location (List[float]): List of values for the 'epsilon_location' parameter.
        epsilon_scale (List[float]): List of values for the 'epsilon_scale' parameter.
        nu_location (List[float]): List of values for the 'nu_location' parameter.
        nu_scale (List[float]): List of values for the 'nu_scale' parameter.
        change_factor_attract (List[float]): List of values for the 'change_factor_attract' parameter.
    """

    __slots__ = ['N', 'alpha', 'tau', 'outdeg_l', 'outdeg_h', 'beta_attr_high',
                 'beta_exploit_low', 'beta_explor_low', 'epsilon_location',
                 'epsilon_scale', 'nu_location', 'nu_scale', 'change_factor_attract']

    def __init__(self, **params: Dict[str, List[Union[int, float]]]):
        
        """Initialize the ABMSweeps instance with parameter values.
        
           Parameters:
           **params: Keyword arguments specifying parameter values as lists.
                The keyword should match the name of a parameter attribute in the class,
                and the value should be a list of parameter values.
                
           Raises:
           AttributeError: If a parameter specified in __slots__ is missing in the input.
        """
        for key, value in params.items():
            setattr(self, key, value)
        #check if all slots have been specified as k-v pairs
        for k in self.__slots__:
            if not hasattr(self, k):
                raise AttributeError(f"You missed the {k} parameter")

    def set_up_grid(self) -> List[Dict[str, Union[int, float]]]:
        
        """Generates a grid of parameter combinations for ABM sweeps.
        
           Returns:
           List[Dict[str, Union[int, float]]]: A list of dictionaries, each representing
           a combination of parameter values for an ABM run.
        """
        
        ABM_params_grid = {k: getattr(self, k) for k in self.__slots__}
        return list(ParameterGrid(ABM_params_grid))

    def grid_search(self, repetitions: int, t: int, T: int, stopping_condition: float = float('inf'), 
                    compute_all: bool = False, path_ts: Optional[str] = None):

        """Perform grid search for parameter combinations and execute network evolution simulations.
           It calls the NetworkEvolution class for each parameter combination and saves the resulting 
           edgelists and parameter information.

           Parameters:
           repetitions (int): Number of repetitions for each parameter combination.
           t (int): Starting time step for the simulation.
           T (int): Ending time step for the simulation.
           stopping_condition (float, optional): Stopping condition for the simulation. Defaults to float('inf').
           compute_all (bool, optional): Flag to compute all statistics for each run. Defaults to False.
           path_ts (str, optional): Directory path to save the results. Defaults to None.

           Returns:
           None

           Note:
           The results are saved in subdirectories named 'param_vectorX' where X is the index of
           the parameter combination in the grid. Each subdirectory contains CSV files with edgelists
           and a txt file ('parameters_dict.txt') with the parameters used for that specific run.
        """

        grid_to_search = self.set_up_grid()
        runs_to_execute: List[NetworkEvolution] = [NetworkEvolution(**g) for g in grid_to_search]
        for idx, run in enumerate(runs_to_execute):
            run.set_types()
            results_to_write: List[List[Tuple[int, int]]] = run.sweeps(repetitions, t, T, stopping_condition, idx, compute_all, path_ts)
            root_path: str = os.getcwd() if path_ts is None else path_ts
            final_path: str = os.path.join(root_path, f'param_vector{idx}')
            os.mkdir(final_path)       
            header: List[str] =  ["From", "To"]
            for file_num, edgelist in enumerate(results_to_write):
                writing_path = os.path.join(final_path, f"{file_num:03}.csv")
                with open(writing_path, 'w', newline='') as file:
                    writer: Any = csv.writer(file, delimiter=',')
                    writer.writerow(header)
                    writer.writerows(edgelist)
            txt_file: str = os.path.join(final_path, "parameters_dict.txt")
            dict_to_write: Dict[str, Union[int, float]] = grid_to_search[idx]
            txt_content: List[str] = ['**Readme: Parameters used to generate the networks in this directory**\n\n']
            txt_content.extend([f"{k} = {v}\n\n" for k, v in dict_to_write.items()])
            with open(txt_file, 'w', newline = '') as file:
                file.writelines(txt_content)

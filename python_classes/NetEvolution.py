import igraph as ig
import numpy as np
import os
import random as random
from random import randrange
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import multiprocessing
import csv
import statistics as stats
from joblib import Parallel, delayed
from typing import List, Dict, Union, Tuple, Optional, Any

class NetworkEvolution:
    def __init__ (self, N: int, alpha: float, tau: int, outdeg_l: float = -1, outdeg_h: float = -1,
                 beta_attr_high: float = 1, beta_exploit_low: float = 1, beta_explor_low: float = 1,
                 epsilon_location: float = 0, epsilon_scale: float = 0.1,
                 nu_location: float = 0, nu_scale: float = 0.1, change_factor_attract: float = 1):
                     
        """Initialize a NetworkEvolution instance.

           Parameters:
           - N (int): The number of agents (nodes) in the network.
           - alpha (float): A parameter controlling the proportion of high-skilled agents.
           - tau (int): The threshold value for high-skilled agents.
           - outdeg_l (float, optional): Low-skilled agents' preferences for out-degree.
           - outdeg_h (float, optional): High-skilled agents' preferences for out-degree.
           - beta_attr_high (float, optional): Attraction preferences towards high-skilled agents, for high- and low-skilled.
           - beta_exploit_low (float, optional): Exploitation preferences for low-skilled agents.
           - beta_explor_low (float, optional): Exploration preferences for low-skilled agents.
           - epsilon_location (float, optional): Location parameter for Gumbel shock for low-skilled agents.
           - epsilon_scale (float, optional): Scale parameter for Gumbel shock for low-skilled agents.
           - nu_location (float, optional): Location parameter for Gumbel shock for high-skilled agents.
           - nu_scale (float, optional): Scale parameter for Gumbel shock for high-skilled agents.
           - change_factor_attract (float, optional): A factor modifying attraction for high-skilled agents towards other high-skilled.
        """
        
        if N <= 0 or not isinstance(N, int):
            raise ValueError("N, the number of nodes, must be a positive integer")
        self.N = N

        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be a float between zero and one")
        self.alpha = alpha

        if tau <= 0 or not isinstance(tau, int):
            raise ValueError("Tau, the threshold, must be a positive integer")
        self.tau = tau

        if change_factor_attract <= 0:
            raise ValueError("Any change in beta_attract for high-skilled must be strictly positive")

        self.change_factor_attract = change_factor_attract
        self.outdeg_l = outdeg_l
        self.outdeg_h = outdeg_h
        self.beta_attr_high = beta_attr_high
        self.beta_exploit_low = beta_exploit_low
        self.beta_explor_low = beta_explor_low
        self.epsilon_location = epsilon_location
        self.epsilon_scale = epsilon_scale
        self.nu_location = nu_location
        self.nu_scale = nu_scale

    def get_params(self) -> Dict[str, Union[int, float]]:

        """This method gathers the user-specified parameters for the simulation and returns them as a
           dictionary. Each parameter is paired with its corresponding description, and the values can be
           integers or floats, depending on the parameter.
           
           Returns:
           Dict[str, Union[int, float]]: A dictionary with parameter names as keys and their respective
           user-specified values as values.
        """

        params_dict = {
        "N": self.N,
        "Alpha": self.alpha,
        "Tau": self.tau,
        "Outdegree_of_Low": self.outdeg_l,
        "Outdegree_of_High": self.outdeg_h,
        "Attraction_to_High": self.beta_attr_high,
        "Exploitation_Redirecting": self.beta_exploit_low,
        "Exploration_Redirecting": self.beta_explor_low,
        "Location_Gumbel_Shock_L": self.epsilon_location,
        "Scale_Gumbel_Shock_L": self.epsilon_scale,
        "Location_Gumbel_Shock_H": self.nu_location,
        "Scale_Gumbel_Shock_H": self.nu_scale,
        "Factor_Multiplying_Attractiveness_High": self.change_factor_attract
        }
        return params_dict

    def set_types(self) -> Dict[str, int]:
        
        """Compute the number of low and high skilled agents based on the user-specified alpha and N.
        
           Returns:
           Dict[str, int]: A dictionary containing the total number of agents, the number of low- and high-skilled agents,
           with keys 'N', 'Number of Low-Skilled Agents', and 'Number of High-Skilled Agents'.
        """
        
        self.low_index: int = round((1 - self.alpha) * self.N)
        self.high_index: int = self.N - self.low_index
        return {'N': self.N, 'Number of Low-Skilled Agents': self.low_index, 'Number of High-Skilled Agents': self.high_index}

    def description(self) -> str:
        return (f"The Network has {self.N} agents. "
            f"There are {self.low_index} low-skilled agents and "
            f"{self.high_index} high-skilled agents. "
            f"The threshold, for high-skilled only, is {self.tau}.")

    def init_network(self) -> ig.Graph:
        
        """Initialize a directed iGraph object with N nodes, where N is the total number of agents.
           The "Type" attribute is assigned to each agent based on whether they are low- or high-skilled, 
           determined by the values calculated with the set_types() method.
           For high-skilled agents, the "Threshold" attribute is set to the user-defined threshold value (tau).
           For low-skilled agents, the "Threshold" attribute is set to infinity.
           The iGraph object is stored as an instance attribute for further use.
           
           Returns:
           ig.Graph: The initialized iGraph object representing the network.
        """
        
        g = ig.Graph(directed = True)
        g.add_vertices(self.N)
        g.vs["Type"] = ["L"] * self.low_index + ["H"] * self.high_index
        g.vs["Threshold"] = [float('inf')] * self.low_index + [self.tau] * self.high_index
        self.net = g
        return self.net 

    def plot_Gumbel_dist(self, extractions: int, which_type: str = "low") -> None:
        
        """Helper method to visualize the histogram and kernel density estimation (KDE) of Gumbel shocks.
        
           Parameters:
           extractions (int): The number of Gumbel random samples to generate and plot.
           which_type (str, optional): The type of agents for which Gumbel shocks are generated.
                                       Must be either "low" or "high" (default is "low").
                                       
           Raises:
           ValueError: If the 'which_type' parameter is not "low" or "high".   
        """
        
        if which_type not in ["low", "high"]:
            raise ValueError("Invalid value for 'which_type'. It must be either 'low' or 'high'.") 
        
        location = self.epsilon_location if which_type == "low" else self.nu_location
        scale = self.epsilon_scale if which_type == "low" else self.nu_scale
        sns.histplot(data=[np.random.gumbel(location, scale) for _ in range(extractions)], kde=True)
        plt.show()

    def evaluate_agent(self, agent_type: str) -> Tuple[List[float], List[str]]:
        
        """Evaluation function for agents of the specified type.
           This method evaluates potential partners for the selected agent of the given type (low or high-skilled).
           For each potential partner j in the network, the agent weighs the attractiveness towards high-skilled
           against the status cost for the advice seeker. If the link with potential partner j already exists in
           the network, the agent considers deleting it, using the same evaluation function. The results are
           stored in lists that provide both the evaluation scores and the actions taken (add_or_rem).
           
           Parameters:
           agent_type (str): The type of the selected agent, either 'low' or 'high'.
           
           Returns:
           Tuple[List[float], List[str]]: A tuple containing two lists:
                 - List[float]: The evaluation scores for each potential partner.
                 - List[str]: The actions taken for each potential partner ('Add' or 'Del').
        """
        
        evaluation: List[float] = []
        add_or_rem: List[str] = []
        for j in range(self.N):
            action = self.handle_agent_action(self.i, j)
            evaluation.append(self.obj_fun_low(self.i) if agent_type == "low" else self.obj_fun_high(self.i))    
            add_or_rem.append(action)
            self.undo_action(action, self.i, j)
        return evaluation, add_or_rem

    def handle_agent_action(self, sender: int, receiver: int) -> str:
        
        """This method takes the sender and receiver IDs and checks if a link already exists between them in the network.
           If a link exists, it deletes the link and returns 'Del'. Otherwise, it adds a link between them and returns 'Add'.
           
           Parameters:
           sender (int): The ID of the sender agent.
           receiver (int): The ID of the receiver agent.
           
           Returns:
           str: The action performed ('Add' or 'Del').
        """
        
        if (sender, receiver) in self.net.get_edgelist():
            self.net.delete_edges(self.net.get_eid(sender, receiver))
            return "Del"
        else:
            self.net.add_edges([(sender, receiver)])
            return "Add"
        
    def undo_action(self, action:str, sender: int, receiver: int) -> None:
        
        """Undo the agent action (Add or Del) based on the previous evaluation.
           
           This method takes the action ('Add' or 'Del'), the sender ID, and the receiver ID, and undoes the action accordingly.
           If the action is 'Add', it deletes the link between the sender and receiver in the network.
           If the action is 'Del', it adds the link between the sender and receiver in the network.
           
           Parameters:
           action (str): The action to undo ('Add' or 'Del').
           sender (int): The ID of the sender agent.
           receiver (int): The ID of the receiver agent.
        """
        
        if action == "Add":
            self.net.delete_edges(self.net.get_eid(sender, receiver))
        else:
            self.net.add_edges([(sender, receiver)])
        
    def obj_fun_low(self, agent) -> float:
        """Objective function of low-skilled agents. Modular code: ObjFun is easy to change.
           
           Parameters:
           agent: The agent for which to calculate the objective function value.
           
           Returns:
           float: The calculated objective function value for the specified low-skilled agent.
        """
        return self.outdeg_l * self.net.degree(mode="out")[agent] + \
               self.beta_attr_high * self.count_links_with_high() + \
               np.random.gumbel(self.epsilon_location, self.epsilon_scale)

    def obj_fun_high(self, agent) -> float:
        """Objective function of high-skilled agents. Modular code: ObjFun is easy to change.
           
           Parameters:
           agent: The agent for which to calculate the objective function value.
           
           Returns:
           float: The calculated objective function value for the specified high-skilled agent.
        """
        return self.outdeg_h * self.net.degree(mode="out")[agent] + \
               self.change_factor_attract * self.beta_attr_high * self.count_links_with_high() + \
               np.random.gumbel(self.nu_location, self.nu_scale)
   
    def evaluation_low(self) -> Tuple[List[float], List[str]]:
        
        """Wrapper: Evaluation function for low-skilled agents."""
        
        return self.evaluate_agent("low")
        
    def evaluation_high(self) -> Tuple[List[float], List[str]]:
        
        """Wrapper: Evaluation function for high-skilled agents."""
        
        return self.evaluate_agent("high")

    def count_links_with_high(self) -> int:
        
        """Counting advice requests to high-skilled agents (a part of the objective function).
           
           Returns:
           int: The number of links that the current agent has with high-skilled agents.
        """
        return sum(1 for j in self.net.neighbors(self.i, mode="out") if self.net.vs["Type"][j] == "H")

    def do_nothing(self, agent_type: str) -> float:
       
        """Evaluation for the do-nothing option (low-skilled or high-skilled).

           Returns:
           float: Satisfaction with the current personal network structure.
        """
       
        outdegree: int = self.net.degree(self.i, mode="out")
        links_with_high: int = self.count_links_with_high()
        if agent_type == "low":
            return outdegree * self.outdeg_l + self.beta_attr_high * links_with_high
        elif agent_type == "high":
            return outdegree * self.outdeg_h + self.change_factor_attract * self.beta_attr_high * links_with_high

    def count_exploitation_redirect(self) -> int:
        
        """Count complete dyads between decision maker (it's always a Low-skilled in this case) and low-skilled neighbors. 
           Exploitation strategy.
           
           Returns:
           int: The number of complete dyads between the decision maker and its low-skilled neighbors.    
        """
        
        neigh_drop: List[int] = self.net.neighbors(self.drop, mode = "out")
        filtered_neigh_drop: List[int] = [n for n in neigh_drop if self.net.vs["Type"][n] == "L"]
        asym_dyads: List[Tuple[int, int]]  = self.listofEdges([self.drop]*len(filtered_neigh_drop), filtered_neigh_drop)
        return sum(self.net.is_mutual(self.net.get_eids(asym_dyads)))

    def count_exploration_redirect(self) -> int:

        """Count existent transitive ties between the decision maker and its low-skilled neighbors to measure exploration strategy.
        
           Returns:
           int: The number of transitive ties between the decision maker and its low-skilled neighbors.
        """

        sub: ig.Graph = self.net.subgraph(list(range(self.low_index)))
        x_ij_id: List[int] = sub.neighbors(self.drop, mode = "out")
        x_ij: List[Tuple[int, int]] = self.listofEdges([self.drop]*len(x_ij_id), x_ij_id)
        x_jh: List[List[Tuple[int, int]]] = [self.listofEdges([j]*len(sub.neighbors(j, mode = "out")), sub.neighbors(j, mode = "out")) for j in x_ij_id]
        x_jh_flat: List[Tuple[int, int]] = list(itertools.chain.from_iterable(x_jh))
        triads: List[Tuple[int, int]] = [(i[0], j[1]) for i, j in itertools.product(x_ij, x_jh_flat) if i[1] == j[0]]
        return sum([1 if t in x_ij else 0 for t in triads])

    def do_nothing_redirect(self) -> float:
        
        """Do nothing option when redirecting (low-skilled only).
           
           Returns:
           float: Satisfaction with current network for redirecting low-skilled agents.
        """
        
        return self.count_exploitation_redirect() * self.beta_exploit_low + \
               self.outdeg_l * self.net.degree(self.drop, mode = "out") + \
               self.count_exploration_redirect() * self.beta_explor_low

    def evalu_redirect(self) -> Tuple[List[float], List[str]]:

        """New evaluation function for Low-skilled, when threshold has been surpassed for at least one High-skilled advice giver."""

        evaluation_redirect: List[float] = []
        add_or_remove_redirect: List[str] = []
        for j in range(self.low_index):
            action = self.handle_agent_action(self.drop, j)
            evaluation_redirect.append(self.outdeg_l * self.net.degree(mode="out")[self.drop] + \
                                       self.beta_exploit_low * self.count_exploitation_redirect() + \
                                       self.beta_explor_low * self.count_exploration_redirect() + \
                                       np.random.gumbel(self.epsilon_location, self.epsilon_scale))
            add_or_remove_redirect.append(action)
            self.undo_action(action, self.drop, j)         
        return evaluation_redirect, add_or_remove_redirect

    def listofEdges(self, sources: List[int], targets: List[int]) -> List[Tuple[int, int]]:
        return list(zip(sources, targets))

    def link_change(self) -> List[Tuple[int, int]]:

        """Main method for changing (adding, deleting) links or doing nothing based on 
           agents' objective functions and thresholds. This method is the core of the agent-based
           simulation and is responsible for dynamically changing the network links. 
           It employs a modular structure using a set of functions for evaluating agents' actions and 
           updating the network accordingly. The method randomly selects an agent 'self.i' from the 
           network and determines its type (low-skilled or high-skilled) using the 'Type' attribute of 
           the corresponding node in the graph.
           If the agent is of type "L" (low-skilled):
              - The method evaluates the attractiveness and potential actions for all possible partners calling 
                'evaluation_low()'. It then selects the most attractive partner for potential link modification
                and handles redirection of existing links in case most attractive partner is an high-skilled, action
                is to "Add" and its threshold is surpassed. If action is instead "Del", a link will be removed with the
                most attractive partner. Else, nothing will happen (do_nothing prevails). 
           If the agent is of type "H" (high-skilled):
              - The method evaluates the attractiveness and potential actions for all possible partners calling 
                'evaluation_high()'. It then selects the most attractive partner for potential link modification. 
                If action is "Add", a link will be established with the most attractive partner. If action is "Del"
                link will be removed. Else, nothing will happen (do_nothing prevails). 
           
           Returns:
           List[Tuple[int, int]]: The updated list of edges (i.e., edgelist) in the network.
        """

        self.i: int = randrange(self.N)
        type_i: str = self.net.vs["Type"][self.i]

        if type_i == "L":
            attractiveness_low, what_to_do_low = self.evaluation_low()
            attractiveness_low[self.i], what_to_do_low[self.i] = self.do_nothing("low"), "Hold"
            max_index: int = attractiveness_low.index(max(attractiveness_low))
            if what_to_do_low[max_index] == "Add":
                if self.net.degree(max_index, mode = "in") >= self.net.vs["Threshold"][max_index]:
                    neigh: List[int] = [n for n in self.net.neighbors(max_index, mode = "in") if self.net.vs["Type"][n] == "L"]
                    if neigh:
                        neigh_id_to_drop_redirect = self.redirecting_nodes(max_index, neigh)
                        for drop in neigh_id_to_drop_redirect:
                            self.drop: int = drop
                            attractiveness_redirect, what_to_do = self.evalu_redirect()
                            attractiveness_redirect[self.drop], what_to_do[self.drop] = self.do_nothing_redirect(), "Hold"
                            max_index_red: int = attractiveness_redirect.index(max(attractiveness_redirect))
                            if what_to_do[max_index_red] == "Add":
                                self.net.add_edges([(self.drop, max_index_red)])
                            elif what_to_do[max_index_red] == "Del":
                                self.net.delete_edges(self.net.get_eid(self.drop, max_index_red))
                    else: #if no low-skilled advice seeker, just add the link (i, max_index)
                        self.net.add_edges([(self.i, max_index)])
                else: #if threshold of high-skilled is not surpassed
                    self.net.add_edges([(self.i, max_index)])

            elif what_to_do_low[max_index] == "Del":
                self.net.delete_edges(self.net.get_eid(self.i, max_index))

            return self.net.get_edgelist()

        elif type_i == "H":
           attractiveness_high, what_to_do_high = self.evaluation_high()
           attractiveness_high[self.i], what_to_do_high[self.i] = self.do_nothing("high"), "Hold"
           max_index_high: int = attractiveness_high.index(max(attractiveness_high))

           if what_to_do_high[max_index_high] == "Add":
              self.net.add_edges([(self.i, max_index_high)])
           elif what_to_do_high[max_index_high] == "Del":
              self.net.delete_edges(self.net.get_eid(self.i, max_index_high))
               
        return self.net.get_edgelist()

    def redirecting_nodes(self, chosen_partner: int, low_skilled_seekers: List[int]) -> List[int]:
        
        """Identify redirecting low-skilled seekers for a given overloaded chosen_partner, when threshold is surpassed.

           Parameters:
           chosen_partner (int): The ID of the node from which to redirect nodes (the overloaded high-skilled agent).
           low_skilled_seekers (List[int]): Low-skilled advice-seekers of overloaded high-skilled advisor. 

           Returns:
           List[int]: A list of low-skilled seekers redirecting away from chosen_partner.
        """
        to_drop: int = randrange(1, len(low_skilled_seekers)+1)
        redir_nodes: List[int] = list(np.random.choice(low_skilled_seekers, to_drop, replace = False))
        ids: List[Tuple[int, int]] = self.net.get_eids(self.listofEdges(redir_nodes, [chosen_partner]*to_drop))
        self.net.delete_edges(ids)
        return redir_nodes            

    def deg_statistics(self, stat: str) -> float:
        
        """Calculate the average degree statistics for a specific type of agents.
        
           Parameters:
           stat (str): The type of degree statistics to calculate ("indeg high", "indeg low", "outdeg high", "outdeg low").
           
           Returns:
           float: The calculated average degree statistics based on the specified type.
        """
        
        stat_mapping: Dict[str, float] = {
                "indeg high": self.net.vs[self.low_index:self.N].degree(mode="in"),
                "indeg low": self.net.vs[:self.low_index].degree(mode="in"),
                "outdeg high": self.net.vs[self.low_index:self.N].degree(mode="out"),
                "outdeg low": self.net.vs[:self.low_index].degree(mode="out")
                }
        return stats.mean(stat_mapping[stat])

    def heter_ties_low(self) -> int:

        """Count the number of heterogeneous ties (L->H) involving low-skilled agents.

           Returns:
           int: The number of L->H ties.
        """
        
        return sum([self.net.are_connected(l,h) for l in list(range(self.low_index)) for h in list(range(self.low_index, self.N))])

    def heter_ties_high(self) -> int:

        """Count the number of heterogeneous ties (H->L) involving high-skilled agents.

           Returns:
           int: The number of H->L ties.
        """
        
        return sum([self.net.are_connected(h,l) for h in list(range(self.low_index, self.N)) for l in list(range(self.low_index))])

    def loopy(self, t: int, T: int, stopping_condition: Union[int, float] = float('inf'), 
              idx: int = 0, compute_all: bool = False, path_ts: Optional[str] = None) -> List[List[Tuple[int, int]]]:

        """Perform network evolution simulation using the link_change() function for T times or until stopping_condition 
           (number of edges) is met.
           
           Parameters:
           t (int): The current iteration number.
           T (int): The total number of iterations to perform.
           stopping_condition (Union[int, float], optional): The condition to stop the simulation when the number of edges is reached.
           idx (int, optional): An index to label the simulation for data saving purposes.
           compute_all (bool, optional): If True, compute additional network statistics during the simulation.
           path_ts (Optional[str], optional): The path to save the time series statistics CSV files.
           
           Returns:
           List[List[Tuple[int, int]]]: A list of edgelists representing the network's evolution over iterations.
           The outer list contains the edgelists for each iteration, starting from an edgelist of size one (first link)
           and ending at the network at T or when the stopping_condition is met.
        """

        self.init_network()
        edgelists: List[List[Tuple[int, int]]] = []

        if compute_all is False:
            while t < T:
                edgelists.append(self.link_change())
                if len(edgelists[t]) >= stopping_condition:
                    break
                t += 1
            return edgelists

        else:
            time_series_data: List[List[Union[float, int]]] = [[] for _ in range(8)]
            while t < T:
                edgelists.append(self.link_change())
                time_series_data[0].append(self.deg_statistics("indeg low"))
                time_series_data[1].append(self.deg_statistics("indeg high"))
                time_series_data[2].append(self.deg_statistics("outdeg low"))
                time_series_data[3].append(self.deg_statistics("outdeg high"))
                time_series_data[4].append(self.net.subgraph(list(range(self.low_index))).density(loops=False))
                time_series_data[5].append(self.net.subgraph(list(range(self.low_index, self.N))).density(loops=False))
                time_series_data[6].append(self.heter_ties_low())
                time_series_data[7].append(self.heter_ties_high())
                if len(edgelists[t]) >= stopping_condition:
                    break
                t += 1
            self.save_time_series_stats(time_series_data, idx, path_ts)
            del time_series_data[:]

            return edgelists

    def save_time_series_stats(self, container_ts: List[List[Union[float, int]]], idx: int, path_ts: Optional[str] = None) -> None:
         
        """Save time series statistics to CSV files.
        
           This method saves the time series statistics to separate CSV files. The statistics include average
           in-degree and out-degree for low-skilled and high-skilled agents, density for low-skilled and high-skilled
           subgraphs, and heterogeneity of ties for low-skilled and high-skilled agents. The CSV files are saved in
           the specified path or the current working directory if no path is provided.
           
           Parameters:
           container_ts (List[List[Union[float, int]]]): A container with the time series statistics data.
           idx (int): An index to label the simulation.
           path_ts (Optional[str], optional): The path to save the time series statistics CSV files.
        """
        landing_path: str = os.getcwd() if path_ts is None else path_ts
        new_landing_path: str = os.path.join(landing_path, f"{idx}time_series{os.getpid()}{random.random()}")
        os.makedirs(new_landing_path, exist_ok=True)
        for file_num, data in enumerate(container_ts):
            file_path = os.path.join(new_landing_path, f"time_series_stats{file_num}.csv")
            with open(file_path, 'w', newline='') as file:
                writer: Any = csv.writer(file, delimiter=',')
                writer.writerow(data)

    def sweeps(self, repetitions: int, t: int, T: int, stopping_condition: Union[int, float] = float('inf'), 
               idx: int = 0, compute_all: bool = False, path_ts: Optional[str] = None) -> List[List[Tuple[int, int]]]:

        """Calling loopy() a number of times specified by the user in repetitions. Parallel computation.
           Gets num_cores and parallelizes each repetition on a core and then reconstruct the results. The "ABMSweeps"
           module will call sweeps() from instances of this class (i.e., NetworkEvolution) for a whole grid of parameters
           (the whole space given by Cartesian product of vectors of parameters). sweeps() here just works for
           one vector of parameters.

           Parameters:
           repetitions (int): The number of loopy() calls.
           From now on, here we have loopy() parameters.
           t (int): The current iteration number for each repetition in the simulation.
           T (int): The total number of iterations to perform in each repetition.
           stopping_condition (Union[int, float], optional): The condition to stop each simulation repetition when the
           number of edges is reached.
           idx (int, optional): An index to label the simulations for data saving purposes.
           compute_all (bool, optional): If True, compute additional network statistics during each simulation repetition.
           path_ts (Optional[str], optional): The path to save the time series statistics CSV files for each simulation.

           Returns:
           List[List[Tuple[int, int]]]: a list of edgelists. Outer-list length will be equal to repetitions.
           Inner lists will be edgelists representing networks obtained by the parallel loopy() workers 
           at T or when stopping_condition is met.
        """
        num_cores: int = multiprocessing.cpu_count()

        loopy_args = (t, T, stopping_condition, idx, compute_all, path_ts) if compute_all else (t, T, stopping_condition, compute_all)
        reps = Parallel(n_jobs= num_cores, verbose=10)(delayed(self.loopy)(*loopy_args) for _ in range(repetitions))
        return [rep[-1] for rep in reps]

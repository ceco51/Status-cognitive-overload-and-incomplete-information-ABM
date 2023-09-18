# Model Documentation with Code-Snippets

**`NetEvolution.py`**

To understand the model in its inner workings "pedagogically", we should follow these steps.

First, download `NetEvolution.py` with all the associated libraries versions that are listed in file `requirements.txt`. Place them in the same folder/create an ad-hoc python virtual environment. Import `NetEvolution.py` module and create a `NetEvolution` object by specifying the relevant model parameters. The positional order of parameters is:  $N$, $\alpha$, $\tau$, $\beta_{0}^{l}$, $\beta_{0}^{h}$, $\beta_{attract}$, $\beta_{EL}^{l}$, $\beta_{ER}^{l}$, $\epsilon$ (location and scale), $\nu$ (location and scale) and $change$ $factor$ $attract$. The class thus assumes that the first parameter is $N$, the number of agents, while the last parameter is the (eventual) change in attractiveness towards high-skilled agents by other high-skilled agents. All parameters except $N$, $\alpha$ and $\tau$ have default values. Therefore, the script will issue a `TypeError` if any of these 3 positional arguments are missing. The default value for $\beta_{0}^{l}$ and $\beta_{0}^{h}$ is $-1$, for $\beta_{attract}$, $\beta_{EL}^{l}$ and $\beta_{EL}^{l}$ is $1$, while the user has the option to initialize two different Gumbel disturbances for $l$ and $h$ respectively ($\epsilon$, $\nu$), whose default location and scale are $0$ and $0.1$. The eventual $change$ $factor$ $attract$ i.e., the eventual shrinkage/increase of $\beta_{attract}$ for high-skilled compared to the baseline for low-skilled ($\beta_{attract}$) has default value of 1. This implies that $\beta_{attract}^{h} = \beta_{attract}^{l}$ i.e., in the default settings high- and low-skilled agents feel the same attractiveness towards other high-skilled agents. In regard to $\epsilon$ and $\nu$, we recommend using the same disturbance values for both, in case the user does not have a valid reasons for doing otherwise. 

```python
import NetEvolution as net

obj = net.NetworkEvolution(20, 0.2, 5, -1, -2, 2.5, 1.5, 1.5, 0, 1, 0, 1, 1)
```

After having created object `obj`, we can call the helper method `get_params()` to retrieve the values of the parameters we used when creating `obj`, to increase transparency. These values are returned in a dictionary, whose keys are substantive short descriptions of the parameters and whose values are our inputs or default values. In addition, we can call the `plot_Gumbel_dist()` method to examine the shapes of the disturbances distributions and plot them as KDE over a histogram. `plot_Gumbel_dist()` has two attributes: `extractions` and `which_type`. `extractions` has no default values and is an integer that tells the program how many draws to extract from the Gumbel distribution of disturbances, while `which_type`, whose default value is "low", is an indicator that tells the program which disturbances distribution to plot (either low- or high-skiled). Remember that the scale and location for the Gumbel disturbances are denoted by $\epsilon$ for low-skilled agents and by $\nu$ for high-skilled ones. If $\epsilon$ and $\nu$ were the same, nothing would change.

```python
obj.get_params()

which_disturbance = "low" #whether we want the disturbance for Low (epsilon) or High-skilled (nu)                          
numb_extractions = 10000 
obj.plot_Gumbel_dist(numb_extractions, which_disturbance)
```
After some checking of our user-defined parameters, we are ready to assign the agents to either the low-skilled group or the high-skilled group by calling the `set_types()` method, using $\alpha$ to make the assignement.

```python
obj.set_types()
```

At this point, the initialization is complete. Initialization of an empty directed network will be automatically performed by the methods used to run the main simulation engine.

Therefore, we are now ready to run the main model Algorithm. We can decide to run it for $T$ periods or $P$-times in parallel for $T$ periods. In both cases, we can set a *stopping condition*, that is we can halt our simulation engine when the number of generated edges hits a particular target before having finished our $T$ periods. In establishing the stopping condition, consider that the maximum number of edges we can observe in a directed network with $N$ nodes is $N \times (N-1)$. Thus, if we set a stopping condition greater than $N \times (N-1)$, it is as if we did not set it at all! In fact, the default value for the stopping condition is infinity, which means that there is practically no stopping condition.

In addition, in both cases we can decide whether we want to save the time-series of network metrics computed for the generated networks in each period as **.csv** files. These metrics are: average in- and out-degrees of both $h$ and $l$, densities among the induced sub-graphs by the two vertex-sets of $h$ and $l$ and number of $h$-to-$l$ and $l$-to-$h$ ties over-time. If we want to save them, we need to specify `compute_all = True` in the following methods.

In case we do not want to save the .csv of time-series network metrics, and we do not want a stopping condition, we can just run:

```python
t = 0 #start tick (default)
T = 50 #number of model runs (the model will run for 50 periods)

obj.loopy(t, T)
```
If we want to add a stopping condition, we have to simply run: 

```python
t = 0 
T = 50

stopping_condition = 20 #simulations stop when reaching 20 edges

obj.loopy(t, T, stopping_condition)
```

If we want to save time-series metrics, we just run the following snippet (we just show it for the case in which we have a stopping condition). Remember to set `compute_all = True` (the default value is `False`, that's why we didn't explicitly set it in previous snippets). If you want to save these time series in a particular folder you just have to specify in the `loopy()` method the argument `path_ts = YOUR_FOLDER`, where YOUR_FOLDER is the path for the particular folder. Otherwise, time series will be saved in the current working directory.   

```python
folder_time_series = '' #place here YOUR folder

t = 0 
T = 50

stopping_condition = 20

obj.loopy(t, T, stopping_condition, compute_all = True, path_ts = folder_time_series)
```


```python
#compute time-series and save them in CURRENT working directory

t = 0 
T = 50

stopping_condition = 20

obj.loopy(t, T, stopping_condition, compute_all = True)
```
The outcome of the `loopy()` method is just a single simulated edgelist under the specified parameters combination. If we want to run `loopy()` $P$-times, with $P \geq 2$, with the same parameters, we should use the `sweeps()` method instead. `sweeps()` parallelizes the computation by using the maximum available number of cores of the machine which is executing the script. The basic usage of `sweeps()` is as follows:

```python
repetitions = 10 #number of parallel runs of loopy() i.e., main simulation engine

t = 0 
T = 50

#no stopping condition, no saving time-series
obj.sweeps(repetitions, t, T)
```
With a stopping condition and no writing (saving) of .csv files:

```python
repetitions = 10 

t = 0 
T = 50
stopping_condition = 20

obj.sweeps(repetitions, t, T, stopping_condition)
```
If we want to save time series, we run a similar code-snippet as `loopy()` case: 

```python
repetitions = 10

t = 0 
T = 50 
stopping_condition = 20
path_ = '' #your output directory

obj.sweeps(repetitions, t, T, stopping_condition, compute_all = True, path_ts = path_)
```
If we want to save .csv in current working directory, do not specify `path_ts`as before:

```python
repetitions = 10

t = 0 
T = 50 
stopping_condition = 20

obj.sweeps(repetitions, t, T, stopping_condition, compute_all = True)
```
Please note that `NetEvolution.py` has not a built-in function to write and save generated edgelists as .csv files. By design, we have built this capability into `gridSearch.py`, which is a wrapper object allowing us to perform computational experiments with different parameter combinations. Nevertheless, since the data structures generated by `loopy()` and `sweeps()` are edgelists (technically, lists of tuples), users can easily write .csv files from them.  

**`gridSearch.py`**

With this second module, we can perform computational experiments with different parameter combinations. As before, we can specify a stopping condition and whether to compute the time series for the network metrics and where to store them. To perform computational experiments by an exhaustive grid search over all combination of parameters, create a file called, for instance, `experiment.py` and write the following: 

```python
import gridSearch as gr
import os

os.chdir('') #place here YOUR directory, where you will see the edgelists

#Define combinations of parameters to investigate as a dictionary, where key:param_name, value:list of values to explore 

exp = gr.ABMSweeps(**{'N': [71], 'alpha': [0.30, 0.70], 'tau':[15], 'outdeg_l':[-1], 'outdeg_h':[-3],
                       'beta_attr_high':[2.5], 'beta_exploit_low':[1.0], 'beta_explor_low':[0.3, 0.8],
                       'epsilon_location':[0], 'epsilon_scale':[0.3], 'nu_location':[0], 'nu_scale':[0.3], 
                       'change_factor_attract':[1]})

#Set up grid of experiment; grid = Cartesian product of parameters 
grid_exp = exp.set_up_grid()

nets_to_generate = 100 #for each parameters' combination we generate 100 networks 
t = 0
T = 200
path_where_to_save_ts = '' #place here YOUR directory path, in which you want to write time series in .csv
edges_to_stop = 20

results_exp = exp.grid_search(nets_to_generate, t, T, stopping_condition = edges_to_stop, compute_all = True, path_ts = 
path_where_to_save_ts) 
```

Subsequently, launch `experiment.py` and collect results in the directories you have specified or in the current working directory.

Note that parameters are passed as key-value pairs, where the key name must exactly match the names we specified, and their values are given as lists, even if we want to examine only one value for a given parameter. The user cannot pass more or fewer arguments. In both cases, the script prints an error message and tells the user which form of the dictionary is appropriate for the class. For example, suppose the user forgets to specify the percentage of agents that are high-skilled. The program issues the following error message: `AttributeError: You missed the alpha parameter`.

If we do not want a stopping condition, we can just avoid to specify it (default will be infinity i.e., will never stop prior to T):

```python
# [...] same as before 

results_exp = exp.grid_search(nets_to_generate, t, T, compute_all = True, path_ts =  path_where_to_save_ts) 
```

When doing computational experiments with `gridSearch`, if we do not specify the path in which to save time-series .csv files the program will use as default option the current directory. Here we show the case of no stopping condition and no specific time-series path specified: 

```python
# [...] same as before 

results_exp = exp.grid_search(nets_to_generate, t, T, compute_all = True)
```

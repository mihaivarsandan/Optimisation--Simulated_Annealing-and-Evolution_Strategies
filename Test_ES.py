from Function import Common
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from Evolution_Strategies import EvolutionStrategies
from Optimisation_ES import optimize_ES_parameters,measure_effectiveness,measure_efficiency
from Simulated_Annealing import SimulatedAnnealing
from Optimise_SA import optimize_SA_parameters

# Optimize ES

with open("optimization_ES.txt", 'a') as file:
    for dim, ratio in [(5, 0.0001)]:
        #optimize_ES_parameters(file, dim=dim, min_ratio=ratio, cov=False, elitist=False)
        optimize_ES_parameters(file, dim=dim, min_ratio=ratio, cov=True, elitist=False)
        #optimize_ES_parameters(file, dim=dim, min_ratio=ratio, cov=False, elitist=True)
        #optimize_ES_parameters(file, dim=dim, min_ratio=ratio, cov=True, elitist=True)

# -------------------STAGE 2----------------------
sa = SimulatedAnnealing(dim=5, trials_max=1125, temp_adapt=True,
                        random_restart=False)
es = EvolutionStrategies(dim=5, population_size=966, cov=False, elitist=True)

measure_efficiency(sa, es, no_runs=10, folder_fname='Comp')
measure_effectiveness(sa, es, no_runs=200)

# -------------------STAGE 3----------------------

es2 = EvolutionStrategies(dim=2, population_size=140, cov=False, elitist=True)

print("ES2", es2.run(seed=27))
es2.visualise_points()
es2.visualise_evaluations()
#es2.multirun_histogram(no_runs=50)
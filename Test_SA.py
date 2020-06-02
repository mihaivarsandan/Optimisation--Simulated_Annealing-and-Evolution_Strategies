from Function import Common
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from Simulated_Annealing import SimulatedAnnealing
from Optimise_SA import optimize_SA_parameters

# -------------------STAGE 1----------------------

# Optimize SA
"""
with open("optimization_SA.txt", 'a') as file:
    for dim in [2, 5]:
        optimize_SA_parameters(file, dim=dim, temp_adapt=False,
                               random_restart=False)
        optimize_SA_parameters(file, dim=dim, temp_adapt=True,
                               random_restart=False)
        optimize_SA_parameters(file, dim=dim, temp_adapt=False,
                               random_restart=True)
        optimize_SA_parameters(file, dim=dim, temp_adapt=True,
                               random_restart=True)



# -------------------STAGE 2----------------------

# Run with the optimized 5D parameters (change manually)
sa = SimulatedAnnealing(dim=5, trials_max=1250, temp_adapt=True,
                        random_restart=False)

print("SA reliability")
print(sa.measure_reliability(no_runs=100, max_error=15*5))

"""
# -------------------STAGE 3----------------------

# Run with the optimized 2D parameters (change manually)
sa2 = SimulatedAnnealing(dim=2, trials_max=75, temp_adapt=True,
                         random_restart=True)

print("SA2", sa2.run(seed=27))
sa2.visualise_points()
sa2.visualise_evaluations()
sa2.multirun_histogram(no_runs=50)




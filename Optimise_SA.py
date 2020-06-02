import datetime as dt
import matplotlib.pyplot as plt
from os.path import join as pjoin
from Simulated_Annealing import SimulatedAnnealing

def optimize_SA_parameters(file, dim=5, max_error=15, no_runs=30,
                           temp_adapt=True, random_restart=True):

    # Write the header of the optimization run
    file.write("------{}-------------------------\r\n".format(
        dt.datetime.now()))
    file.write("{0} dimensions; using adaptive temp: ".format(dim) +
               "{1}; random restarts: {2}\r\n"
               .format(dim, temp_adapt, random_restart))

    # Initialise optimization parameters
    best_ratio = 0
    best_gl_mean = float("inf")
    best_parameter = -1

    for trials_max in range(25, 625, 25):
        print(trials_max)
        sa = SimulatedAnnealing(dim=dim, trials_max=trials_max*dim,
                                random_restart=random_restart,
                                temp_adapt=temp_adapt)
        ratio, gl_mean, *_ = sa.measure_reliability(
            no_runs=no_runs, max_error=0.2*dim)
        print(gl_mean)
        # Document the results with different parameter values
        file.write("{0} --:-- {1}, {2}\r\n"
                   .format(trials_max*dim, ratio, gl_mean))
        if ((ratio > best_ratio) or
           (ratio == best_ratio and gl_mean < best_gl_mean)):
            best_ratio = ratio
            best_gl_mean = gl_mean
            best_parameter = trials_max*dim
            file.write(" -- This is the new best\r\n")

    # Report the best parameter set
    file.write("Best parameters: {}\r\n".format(best_parameter))
    file.write("Solution ratio and global minima mean: {0}, {1}\r\n"
               .format(best_ratio, best_gl_mean))
    file.write("---------------------------------------------------------\r\n")
    # Return the best population size
    return best_parameter


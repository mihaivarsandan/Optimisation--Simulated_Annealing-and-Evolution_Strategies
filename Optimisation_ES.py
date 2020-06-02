import datetime as dt
import matplotlib.pyplot as plt
from os.path import join as pjoin
from Evolution_Strategies import EvolutionStrategies
import numpy as np


def optimize_ES_parameters(file, dim=5, min_ratio=0.6, max_error=15,
                           no_runs=30, cov=True, elitist=False):
    # Write the header of the optimization run
    file.write("------{}-------------------------\r\n".format(
        dt.datetime.now()))
    file.write("{0} dimensions; using rotations: {1}; elitist: {2}\r\n"
               .format(dim, cov, elitist))
    # Initialise optimization parameters
    earliest_stop = 10001
    best_parameters = (-1, -1)
    

    for population_size in range(1176,2296,224):
        print(population_size)
        # Initialize algorithm parameters
        es = EvolutionStrategies(dim=dim, population_size=population_size,
                                 cov=cov, elitist=elitist, _stop=True)
        # Do a number of runs and get the time average and convergence ratio
        stop_time, global_counts = 0, 0
        global_min = []
        for i in range(no_runs):
            es.reset()
            # If stopped earlier, returns (run_time, None)
            time, check = es.run(seed=i*10)
            stop_time += time if check is None else 10000
            if dim==2 and es.best_value<-24.03:
                global_counts +=1
                global_min.append(es.best_value)
            elif dim==5 and es.best_value<-57.5:
                global_counts +=1
                global_min.append(es.best_value)
        ratio = global_counts/no_runs
        print(ratio)
        stop_time /= no_runs
        if len(global_min) == 0:
            global_mean = 0
        else:
            global_mean = np.mean(global_min)
        # Document the results with different parameter values
        file.write("{0} --:-- {1}, {2}, {3}\r\n"
                   .format(population_size, ratio, stop_time,global_mean))
        if (ratio < min_ratio):
            file.write(" -- Ratio too small\r\n")
        elif (stop_time < earliest_stop):
            earliest_stop = stop_time
            best_parameters = (population_size*dim, ratio)
            file.write(" -- This is the new best\r\n")

    # Report the best parameter set
    file.write("Best parameters: {}\r\n".format(best_parameters[0]))
    file.write("Solution ratio: {}\r\n".format(best_parameters[1]))
    file.write("---------------------------------------------------------\r\n")
    # Return the best population size
    return best_parameters[0]


def measure_efficiency(sa, es, no_runs=10, folder_fname='.'):
    """Measure the efficiency by plotting best found value throughout the run.

    Parameters
    ----------
    sa : SimulatedAnnealing - an instance of SA optimizer with set parameters
    es : EvolutionStrategies - an instance of ES optimizer with set parameters
    no_runs : int - number of runs to do (each provides an image)
    floder_fname : str - name of an EXISTING foler where to save the images

    Returns
    -------
    None

    """
    if sa.dimension != es.dimension:
        raise ValueError('Both optimisers must have the same dimensions')

    for i in range(no_runs):
        sa.reset()
        es.reset()
        print(sa.run(seed=i*10))
        print(es.run(seed=i*10))
        plt.plot(range(1, len(sa.best_archive)+1), sa.best_archive,
                 'r', label='SA')
        plt.plot(range(1, len(es.best_archive)+1), es.best_archive,
                 'b', label='ES')
        true_min = [-60] * 10000
        plt.plot(range(1, 10001), true_min, 'k--', label='True')
        ax = plt.gca()
        ax.set_xlim([1, 10000])
        #ax.set_title("SA vs ES efficiency")
        plt.legend(loc='upper right')
        plt.ylabel("Function value")
        plt.xlabel("Evaluation number")
        plt.savefig(pjoin(folder_fname, "fig{}".format(i)))
        plt.clf()


def measure_effectiveness(sa, es, no_runs=25):
    """Measure effectiveness by plotting values found in decreasing order.

    Parameters
    ----------
    sa : SimulatedAnnealing - an instance of SA optimizer with set parameters
    es : EvolutionStrategies - an instance of ES optimizer with set parameters
    no_runs : int - number of runs to do

    Returns
    -------
    None

    """
    es_values = []
    sa_values = []
    for i in range(no_runs):
        print(i)
        es.reset()
        sa.reset()
        es_value, _ = es.run(seed=i)
        sa_value, _ = sa.run(seed=i)
        es_values.append(es_value)
        sa_values.append(sa_value)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    es_values.sort(reverse=True)
    sa_values.sort(reverse=True)
    ax.plot(range(no_runs), sa_values, 'r', label='SA')
    ax.plot(range(no_runs), es_values, 'b', label='ES')
    true_min = [-60] * no_runs
    ax.plot(range(no_runs), true_min, 'k--', label='True')
    #ax.set_title("Best values of 100 runs in decreasing order")
    plt.legend(loc='upper right')
    plt.ylabel("Value")
    plt.show()
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits import mplot3d

class Common(object):
    



    def __init__(self):
        """Initialise the class."""
        self.evaluations = []
        self.best_archive = []
    

    def check_solution(self,x):

        if len(x) != 5 and len(x) !=2:
            raise ValueError('Input is not of correct dimension: {}'.Format(len(x)))
        return (x >= -2).all() and (x <= 2).all()

    def func(self,x):
        if not self.check_solution(x):
            return None

        val = np.arange(1,6)
        value = 0
        for j in val:
            value += -np.sum(j*np.sin((j+1)*x + j*np.ones(x.shape)  ))
        self.evaluations.append(value)
        self.best_archive.append(self.best_value)
        return value
    


    def plot_func_contour(self,mod="contour"):
        """Plot the contours of the function.

        Parameters
        ----------
        None

        Returns
        -------
        matplotlib.figure.Figure - reference to the figure
        matplotlib.axes.Axes - references to the axes of the figure

        """
        X = np.linspace(-2, 2, 501)
        # Y needs to be inverse because image highest value of y should be at
        # the top of the image
        Y = np.linspace(2, -2, 501)
        Z = np.empty((len(Y), len(X)))
        for j, x in enumerate(X):
            for i, y in enumerate(Y):
                Z[i, j] = self.func(np.array([x, y]))


        fig, ax = plt.subplots()
        im = plt.imshow(Z, cmap=cm.seismic, vmin=Z.min(), vmax=Z.max(),
                extent=[-2, 2, -2, 2])
        im.set_interpolation('bilinear')
        fig.colorbar(im)
        return fig, ax

    def plot_func_3D(self):
        X = np.linspace(-2, 2, 101)
        Y = np.linspace(2, -2, 101)
        Z = np.empty((len(Y), len(X)))
        """
        for  x in X:
            for y in Y:
                z = self.func(np.array([x, y]))
                Z.append(z)
        """
        for j, x in enumerate(X):
            for i, y in enumerate(Y):
                Z[i, j] = self.func(np.array([x, y]))

        X_mesh,Y_mesh = np.meshgrid(X,Y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X_mesh, Y_mesh, Z, 50, cmap='seismic')
        ax.set_zlabel('$f$')
        ax.view_init(-50, -50)

        return fig,ax

    def visualise_points(self):
        """Plot all points that have been explored."""
        fig, ax = self.plot_func_contour()
        num = 0 
        for Points in self._archive:
            num += 1
            if num==15:
                point = np.array(Points)
                plt.scatter(point[:, 0], point[:, 1],c='black')
            
            
        
        print("Number of kids")
        print(num)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        if (self.__class__.__name__ == "EvolutionStrategies"):
            name = "ES"
        else:
            name = "SA"
        #ax.set_title("{} accepted/selected solutions".format(name))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

    def visualise_evaluations(self):
        """Plot the value of evaluations made."""
        plt.plot(range(1, len(self.evaluations)+1), self.evaluations)
        ax = plt.gca()
        #ax.set_title("{} efficiency".format(self.__class__.__name__))
        ax.set_ylim(bottom=-15*self.dimension)
        ax.set_xlim(right= 10000, left= 0 )
        plt.ylabel("Function value")
        plt.xlabel("Evaluation number")
        plt.show()

    def multirun_histogram(self, no_runs=25, bins=50):
        """Plot a histogram of best values found during multiple runs.

        Parameters
        ----------
        no_runs : int - number of runs to do
        bins : int/array - number of bins to use or an array of bin edges

        Returns
        -------
        None

        """
        values = []
        for i in range(no_runs):
            self.reset()
            value, solution = self.run(seed=i*10)
            values.append(value)
        plt.hist(values, bins=bins)
        ax = plt.gca()
        ax.set_title("Histogram of {} best values"
                     .format(self.__class__.__name__))
        plt.xlabel("Best value")
        plt.show()

    def measure_reliability(self, no_runs=50, max_error=0.2):
        """Measure the number of global convergences and statistical info.

        Parameters
        ----------
        no_runs : int - number of runs to do
        max_error : int/float - global convergence is counted if the best
                                value is within +max_error*dimension

        Returns
        -------
        float - ratio of global convergences to number of runs
        float - mean of the global convergence values
        float - std of the global convergence values
        float - mean of all convergence values
        float - std of all convergence values

        """
        local_min = []
        global_min = []
        for i in range(no_runs):
            self.reset()
            value, solution = self.run(seed=i*10)
            if self.dimension == 2 and value <-24.03:
                global_min.append(value)
            elif self.dimension == 5 and value<-57.8 + max_error:
                global_min.append(value)
            else:
                local_min.append(value)
        # Calculate measurements
        ratio = len(global_min)/no_runs
        if len(global_min) == 0:
            global_mean, global_std = 0, float("inf")
        else:
            global_mean = np.mean(global_min)
            global_std = np.std(global_min)
        total_mean = np.mean(global_min + local_min)
        total_std = np.std(global_min + local_min)
        return ratio, global_mean, global_std, total_mean, total_std




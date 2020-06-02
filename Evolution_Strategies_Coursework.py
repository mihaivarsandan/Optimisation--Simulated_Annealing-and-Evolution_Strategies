import numpy as np
from Function import Common

class EvolutionStrategies(Common):
    """
    Contains all the requirements to run an Evolution Strategies
    Optimisation with different strategies:
    -rotation strategy
    -selection strategy
    """
 

    def __init__(self, dim=5, population_size=1000, rot=True, elitist=False,
                 stop=False):
        super(EvolutionStrategies, self).__init__()

        """ General Parameters"""
        self.rot, self.elitist = rot, elitist
        self.dimension = dim
        """Selection parameters"""
        self.population_size = population_size
        self.parents_size = int(population_size/7)
        """Mutation parameters"""
        self.tau = 1 / np.sqrt(2*np.sqrt(dim))
        self.tau_dash = 1 / np.sqrt(2*dim)
        self.beta = 0.0873
        
        
        self.best_solution = None
        self.best_value = float("inf")
        self.archive_generation = []
        self.current_generation =[]

        """Stopping criteria"""
        if stop:
            self.stop = 15*dim
        else:
            self.stop = 0

        if rot:
            self.rot_dim = int(dim*(dim-1)/2)

    def run(self, evaluations=10000, seed=None):
        """ Main function that runs the algorithm"""
        
        np.random.seed(seed) # Set the seed for the random generator
        initial = []
        for i in range(self.population_size):
            if self.rot:
                initial.append(np.concatenate(
                    (np.random.uniform(-2, 2, self.dimension),
                     np.random.rand(self.dimension),
                     np.random.uniform(-np.pi, np.pi, self.rot_dim))))
            else:
                initial.append(np.concatenate(
                    (np.random.uniform(-2, 2, self.dimension),
                     np.random.rand(self.dimension))))
        self.current_generation.extend(initial)
        initial_assesments = self.assess(initial)
        parents, parent_assesments, stop = self.select(initial_assesments)
        evals_done = self.population_size
        self.archive_generation.append(self.current_generation)

        """Main loop"""
        while evals <= evaluations-self.population_size:
            self.current_generation=[]
            kids = self.recombine(parents)
            kids = self.mutate(kids)
            self.current_generation.extend(kids)
            kids_assesments = self.assess(kids)
            evals += len(kids)
            self.archive_generation.append(self.current_generation)
            if self.elitist:
                parents, parent_assesments, stop = self.select(
                    kids_assesments+parent_assesments)
            else:
                parents, parent_assesments, stop = self.select(
                    kids_assesments)
           
            if stop:
                return evals_done, None

        return self.best_value, self.best_solution

    def recombine(self, population):
        """Recombination part"""
        new_population = []
        for i in range(self.population_size):
            """Choosing the parents"""
            [i1, i2] = np.random.choice(len(population), 2)
            individual_length = len(population[0])
            kid = np.empty(individual_length)
            """Discrete recombination for control variables"""
            for j in range(self.dimension):
                choice = np.random.randint(2)
                kid[j] = (choice*population[i1][j] +
                          (1-choice)*population[i2][j])
            """Intermediate recombination for strategy variables"""
            for j in range(self.dimension, individual_length):
                choice = np.random.rand()
                kid[j] = (choice*population[i1][j] +
                          (1-choice)*population[i2][j])
            new_population.append(kid)
        return new_population

    def mutate(self, population):
        """Mutation part"""
        new_population = []
        dim = self.dimension
        if self.rot:
            dim2 = self.rot_dim
        
        for a in population:
            n = np.empty(len(a))
            """Standard deviations"""
            n[dim:dim*2] = a[dim:dim*2] * np.exp(
                           self.tau_dash * np.random.normal() +
                           self.tau * np.random.normal(size=dim))
            """Rotations"""
            if self.rot:
                """"Matrix needs to be positive semidefinite for inversal so different methods
                are used to transform the matrix into the required form"""
                for i in range(1000):
                    n[-dim2:] = (a[-dim2:] +
                                 self.beta * np.random.normal(size=dim2))
                    cov = self.covariance_matrix(n)
                    if cov is not None:
                        break
                else:
                    cov = self.covariance_matrix(n, detect_err=False)
            else:
                cov = self.covariance_matrix(n)
            """Control variables"""
            mean = np.zeros(dim)
            n[:dim] = a[:dim] + np.random.multivariate_normal(mean, cov)
            
            new_population.append(n)
        return new_population

    def covariance_matrix(self, solution, detect_err=True):
        """Build a covariance matrix from the solution"""
        dim = self.dimension
        """If no rotation strategy is used, a simply a diagonal with the variance will be built"""
        var = np.square(solution[dim:dim*2])
        if not self.rot:
            return np.diag(var)
        """If the rotation strategy is selected, a symmetric matrix is built"""
        A = np.zeros((dim, dim))
        rot = solution[-self.rot_dim:]
        idx = 0
        for i in range(dim):
            for j in range(i):
                A[i, j] = 0.5 * (var[i] - var[j]) * np.tan(2*rot[idx])
                idx += 1
            A[i, i] = var[i]
        A = (np.maximum(A, A.T))
        """Try to make the matrix positive definite"""
        if detect_err and not np.all(np.linalg.eigvals(A) > 0):
            return None
        return np.linalg.inv(A)

    def assess(self, population):
        """Assement of the value used for the current generation"""
        assessments = []  
        for idx, individual in enumerate(population):
            value = self.func(individual[:self.dimension])
            if value is not None:
                assessments.append((individual, value))
        return assessments

    def select(self, assessments):
        """Selection part of the algorithm"""
        val = sorted(assessments, key=lambda c: c[1])
        if self.stop != 0:
            terminate = abs(val[-1][1] - val[0][1]) <= self.stop
        else:
            terminate = False
    
        if val[0][1] < self.best_value:
            self.best_value = val[0][1]
            self.best_solution = val[0][0][:self.dimension]
        selected = []
        for i in range(self.parents_size):
            if i == len(val): 
                break
            selected.append(val[i][0])
        return selected, val[:self.parents_size], terminate
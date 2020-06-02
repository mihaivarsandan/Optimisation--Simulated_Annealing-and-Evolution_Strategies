import numpy as np
from Function import Common

class SimulatedAnnealing(Common):
 """
    Contains all the requirements to run a Simulated Annealing
    Optimisation with different strategies:
    -temperature adaptive strategy
    -restart strategy
    """

    def __init__(self, dim=5, L_k=1000, temp_adapt=True, random_restart=True):
        super(SimulatedAnnealing, self).__init__()  
        self.temp_adapt, self.random_restart = temp_adapt, random_restart
        self.dimension = dim
        """Generator parameters"""
        self.alpha = 0.1
        self.omega = 2.1
        """Temperature parameters"""
        self.initial_search = 50  # fixed
        self.L_k = L_k
        self.accepted_min = int(L_k*0.6)
        """Restart acceptance ratio"""
        self.ratio = 0.01

        # Algorithm values to save
        self.best_solution = None
        self.best_value = float("inf")
        self.store_accepted_values = []
        self.archive_temp = []
        self.restart = False
        self.end = False
        self.temp_values = []
        self.new_best = False

    def run(self, evaluations=10000, seed=None):
        """ Main function that runs the algorithm"""
        
        np.random.seed(seed) # Set the seed for random generator
        solution = np.random.uniform(-1, 1, self.dimension) #intial solution
        value = self.func(solution*2)
        self.archive_solution(solution, value)
        """Initialisation"""
        D = np.eye(self.dimension)
        temp = self.initial_search(D)
        trials, accepted = 0, 0 # Initialise parameters

        """Start of main loop"""
        for identity in range(self.initial_search, evaluations+1):
            new_solution, delta = self.generate_solution(solution, D)
            new_value = self.func(new_solution*2)
            trials += 1
            if new_value > value: # Check if inceasing solution is accepted
                prob = self.probability(new_value-value, temp, delta)
                if prob >= np.random.rand(1):
                    solution, value = new_solution, new_value
                    D = self.update_generator(D, delta)
                    self.archive_solution(solution, value)
                    accepted += 1
            
            else: # All decreasing solution accepted
                solution, value = new_solution, new_value
                D = self.update_generator(D, delta)
                self.archive_solution(solution, value)
                accepted += 1

            """Check if restart is required"""

            if accepted >= self.accepted_min or trials >= self.L_k:
                if not self.new_best and accepted/trials < self.ratio:
                    if self.random_restart:
                        self.restart = True
                        solution = np.random.uniform(-1, 1, self.dimension)
                    else:
                        
                        self.restart = True
                        solution = self.best_solution/2
                    value = self.func(solution*2)
                    trials, accepted = 0, 0
                    continue  
                temp, trials, accepted = self._update_temperature(temp)

         
        self._archive.append(self._archive_temp)

        return self.best_value, self.best_solution

    def archive_solution(self, solution, value):
        """Update storage"""
        if self.restart:
            self.store_accepted_values.append(self.archive_temp)
            self.restart = False
            self.archive_temp = []

        self.archive_temp.append((solution*2).tolist())

        if value < self.best_value:
            self.best_solution = solution*2
            self.best_value = value
            self.new_best = True
        self.temp_values.append(value)

    def generate_solution(self, x, D):
        """Generate solution"""
        delta = D.dot(np.random.uniform(-1, 1, self.dimension))
        while not self.check_solution((x + delta)*2):
            
            delta = D.dot(np.random.uniform(-1, 1, self.dimension))
        return x + delta, delta

    def update_generator(self, D, delta):
        """Update the solution generator"""
        R = np.diag(abs(delta))
        D = (1-self.alpha) * D + self.alpha * self.omega * R
        return D

    def update_temperature(self, temp):
        """Update the temperature """
        if not self.temp_adapt:
            temp_param = 0.95
        elif len(self.temp_values) < 2:
            temp_param = 0.5 
        else:
            temp_param = max(0.5, np.exp(-0.7*temp/np.std(self.temp_values)))
        self.temp_values = []
        self.new_best = False
        return temp_param*temp, 0, 0

    def probability(self, delta, temp, delta):
        """Calculate the acceptance probability"""
        step_size = np.sqrt(np.sum(np.square(delta)))
        return np.exp(-delta/(temp*step_size))

    def initial_search(self, D):
        """Perform the initial search"""
        solution = np.random.uniform(-1, 1, self.dimension)
        values = [self.func(solution*2)]
        for i in range(self.initial_search):
            solution, delta = self.generate_solution(solution, D)
            D = self.update_generator(D, delta)
            values.append(self.func(solution*2))
        return np.std(values)
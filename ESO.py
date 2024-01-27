import numpy as np
import time

class ESO:
   # Electric Storm Optimization
    def __init__(self, function, constraints, n_rays=50, iterations=10000, objective ='min',  bounds=None, verbose=True):
         # Inicialización con parámetros
        self.cache = {} 
        self.function = function # Objective function
        self.constraints = constraints # Problem constraints
        if n_rays < 5:
            print("Warning: The number of rays can not be less than 5")
            n_rays = 5
        self.n_rays = n_rays # Number of rays to be deployed
        self.initial_rays = n_rays 
        self.ke = 2
        self.explored_solutions = set()  # Set to store explored solutions
        self.iterations = iterations # Number of optimization iterations
        self.iteration_current = 0 # Current iteration
        self.objective = objective # Objective function type
        self.bounds = bounds or [(0, 1) for _ in range(2)] # Search space bounds
        self.dim = len(self.bounds) # Problem dimensionality
        self.verbose = verbose # Whether to print progress information
        self.stagnation = np.zeros(n_rays, dtype=int) # Stagnation counter for each ray
        self.promising_areas = [] # List to storage promising areas
        self.rays = np.array([self.initialize_ray() for _ in range(n_rays)]) # Initialize rays
        self.objective_values = [] # List to storage objective values
        self.iteration_times = []  # List to storage iteration times
        self.best_positions = [] # List to storage best positions
        self.ray_history = [[] for _ in range(n_rays)] # List to storage ray history
        
                   
    def initialize_ray(self): 
        " Initialize the rays on the promising areas if any, otherwise ramdomly within the bounds"
       
        position = None 
        if self.promising_areas:
            # If there are promising areas, choose one and perturb its position slightly
            selected_area = np.random.choice(self.promising_areas)
            position = selected_area['position'] 
        else:
            # Otherwise, initialize a ray randomly within the bounds
            bounds_low, bounds_high = np.array(self.bounds).T
            position = np.random.uniform(bounds_low, bounds_high)

        if self.is_path_valid(position):
            if tuple(position) in self.cache:
                best_score = self.cache[tuple(position)]
            else:
                best_score = self.function(position)
                self.cache[tuple(position)] = best_score
        else:
            best_score = float('inf')
        return {'position': position, 'best_position': np.copy(position), 'best_score': best_score}
    
    def is_path_valid(self, solution):
        " Check whether the solution is valid or not "
        
        return all(constraint(solution) <= 0 if callable(constraint) else True for constraint in self.constraints)
    
    def adjust_path(self, solution):
        " Adjust the solution to ensure that it is valid "
       
        adjusted_solution = solution.copy()
        for _ in range(5):
            if not self.is_path_valid(adjusted_solution):
                direction = np.random.randn(self.dim)
                # step_size = np.random.uniform(0.01, 0.5)   
                step_size = self.cros_points() 
                adjusted_solution += step_size * direction
                adjusted_solution = np.clip(adjusted_solution, *np.array(self.bounds).T)
            else:
                break
        return adjusted_solution
        
    def update_ray_position(self, lightning_position, storm_centers, electric_field_intensity):
        " Update the position of the rays near to the storm centers "
        new_lightning_position = np.zeros_like(lightning_position)
        for storm_center in storm_centers:
            electric_randomness1, electric_randomness2 = np.random.random(), np.random.random()
            electric_attraction = 2 * electric_field_intensity * electric_randomness1 - electric_field_intensity
            electric_force = 2 * electric_randomness2
            electric_displacement = abs(electric_force * storm_center - lightning_position)
            new_lightning_position += storm_center - electric_attraction * electric_displacement

        new_lightning_position /= len(storm_centers)  # Average of updated positions
        return new_lightning_position
    
    def is_new_solution(self, solution):
        solution_tuple = tuple(solution)  # Convert array to tuple for hashability
        if solution_tuple in self.explored_solutions:
            return False
        else:
            self.explored_solutions.add(solution_tuple)
            return True
        
    def field_intensity(self):
        return np.exp2(self.ke - self.iteration_current * (self.ke / self.iterations))
    
    def cros_points(self):
        return (self.field_intensity() - np.exp2(-self.ke)) / (np.exp2(self.ke) - np.exp2(-self.ke))
        

    def branch_and_propagate(self, idx):
        "Propagate rays based on the interaction with others rays and the intensity of the field"
        while True:
            # Reinciar rayo si está estancado
            if self.stagnation[idx] > 10:
                return self.initialize_ray()['position']
            intensity_decay = self.field_intensity() 
            storm_center = [self.rays[i]['position'] for i in range(4)] # 3 best rays
            new_position = self.update_ray_position(self.rays[idx]['position'], storm_center, intensity_decay)
            adapted_path = np.clip(new_position, *np.array(self.bounds).T)
            candidates = np.random.choice([i for i in range(self.n_rays) if i != idx], 4, replace=False)
            base_ray = self.rays[candidates[0]]['position']
            influence_ray1 = self.rays[candidates[1]]['position']
            influence_ray2 = self.rays[candidates[2]]['position']
            influence_ray3 = self.rays[candidates[3]]['position']
            disturbance = intensity_decay * (influence_ray1 - influence_ray2 - influence_ray3)
            new_path = np.clip(base_ray + disturbance, *np.array(self.bounds).T)
            cross_point_prob =  self.cros_points() ## Aqui elimine del inicio de la ecuacion la multiplicacion por 0.9
            cross_points = np.random.rand(self.dim) < cross_point_prob

            if not np.any(cross_points):
                random_index = np.random.randint(0, self.dim)
                adapted_path[random_index] = new_path[random_index]

            if self.is_new_solution(adapted_path):
                break
            # Si la solución ya ha sido explorada, el bucle se repite generando una nueva
        
        return adapted_path
      
    def select_path(self, idx, adapted_path):
        "Evaluate the objective function at the adapted path and update the rays if the objective function is better "
        if tuple(adapted_path) in self.cache:
            adapted_score = self.cache[tuple(adapted_path)]
        else:
            adapted_score = self.function(adapted_path)
            self.cache[tuple(adapted_path)] = adapted_score
        
        if (self.objective == 'min' and adapted_score < self.rays[idx]['best_score']) or \
            (self.objective == 'max' and adapted_score > self.rays[idx]['best_score']):
            self.rays[idx]['position'] = adapted_path
            self.rays[idx]['best_position'] = adapted_path
            self.rays[idx]['best_score'] = adapted_score
            self.stagnation[idx] = 0
        else:
            self.stagnation[idx] += 1
             
    def update_optimal_strike(self, idx, global_best_position, global_best_score):
        "Update the global best position and score of the rays if the objective function is better "
        if (self.objective == 'min' and self.rays[idx]['best_score'] < global_best_score) or \
        (self.objective == 'max' and self.rays[idx]['best_score'] > global_best_score):
            global_best_position = self.rays[idx]['best_position']
            global_best_score = self.rays[idx]['best_score']
            self.update_best_positions_history(self.rays[idx]['best_position'], self.rays[idx]['best_score'])
        return global_best_position, global_best_score
    
    def update_best_positions_history(self, position, score):
        " Update the list of best positions "
        self.best_positions.append((score, position))
        self.best_positions.sort(key=lambda x: x[0], reverse=(self.objective == 'max'))
        self.best_positions = self.best_positions[:5]  # Keep only the 5 best positions
        
    def identify_strong_fields(self):
        "Identify areas of the search space where the solutions are better"
        
        self.promising_areas = []
        percentile_threshold = np.percentile([ray['best_score'] for ray in self.rays], 10)
        self.promising_areas = [
            {'position': ray['best_position'], 'score': ray['best_score']}
            for ray in self.rays if ray['best_score'] < percentile_threshold
        ]
                            
    def evaluate_field_variability(self):
        "Measure the normalized variability of the field in the search space"
        positions = np.array([ray['position'] for ray in self.rays])
        max_diff = np.ptp(positions)  # Peak-to-peak (max - min) difference across all dimensions
        max_diff = max(max_diff, 1e-6)  # Avoid division by zero
        return np.std(positions) / max_diff

    def adapt_storm_conditions(self):
        "Dynamically adjust the number of rays and radius based on normalized field intensity"
       
        # Ajustar el número de rayos: Nunca debe ser menor que 5
        self.n_rays = max(int(np.round(self.initial_rays * self.cros_points())), 5)
        
    def optimize(self):
            "Optimize the search space"
            
            if not callable(self.function) or not all(callable(constraint) for constraint in self.constraints):
                raise ValueError("Function and all constraints must be callable.")

            global_best_position = self.rays[0]['position'].copy()
            global_best_score = self.rays[0]['best_score']

            for iteration in range(self.iterations):
                self.iteration_current = iteration
                iteration_start_time = time.time()
                self.adapt_storm_conditions()

                for idx in range(self.n_rays):
                    adapted_path = self.branch_and_propagate(idx)
                    
                    # Adjust the path if it is not valid
                    self.rays[idx] = self.adjust_path(self.rays[idx])
                    self.select_path(idx, adapted_path)
                    global_best_position, global_best_score = self.update_optimal_strike(idx, global_best_position, global_best_score)
                            
                    # Reinitialize rays that have stagnated
                    if self.stagnation[idx] > 10:
                        self.rays[idx] = self.initialize_ray()
                        self.stagnation[idx] = 0
    
                self.identify_strong_fields()  
       
                iteration_duration = time.time() - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.objective_values.append(global_best_score)
                self.best_positions.append(global_best_position)

                if self.verbose:
                    print(f"Iteration: {iteration+1}, Current best: {self.rays[0]['best_score']:.20e}, Global best: {global_best_score:.20e}, Iteration time: {iteration_duration:.5f} seconds")

            return global_best_position, global_best_score
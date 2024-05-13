import numpy as np
import time

class ESO:
   # Electric Storm Optimization
    def __init__(self, function, pop_size=50, max_iter=1000, max_eval = 500000 ,objective ='min', bounds=None, verbose=True):
            # Inicialización con parámetros
            self.cache = {} # Cache to storage the evaluation to the function
            self.ionized_areas_indices = [] # List to storage ionized areas indices
            self.ionized_areas_positions = [] # List to storage ionized areas positions
            self.storm_power = 0 # Storm's power initial state
            self.bounds = bounds or [(0, 1) for _ in range(2)] # Search space bounds
            self.dim = len(self.bounds) # Problem dimensionality
            self.function = function # Objective function
            self.function_evaluations = 0 # Number of evaluations to the objective function
            self.function_evaluations_list = []
            self.lightning = np.array([self.initialize_lightning() for _ in range(pop_size)]) # Initialize lightning
            self.n_lightning = pop_size # Number of lightning to be deployed
            self.objective = objective # Objective function 
            self.iterations = max_iter # Number of optimization iterations
            self.iteration_current = 0 # Current iteration
            self.field_resistance = 0 # Field's resistance initial state
            self.field_intensity = 0 # Field's intensity initial state
            self.verbose = verbose # Whether to print progress information
            self.stagnation = np.zeros(pop_size, dtype=int) # Stagnation counter for each lightning
            self.objective_values = [] # List to storage objective values
            self.iteration_times = []  # List to storage iteration times
            self.best_positions = [] # List to storage best positions
            self.lightning_history = [[] for _ in range(pop_size)] # List to storage lightning history    
            self.max_evaluations = max_eval  # Maximum number of evaluations       
              
    def identify_ionized_areas(self):
        # Identify high-energy areas based on their performance
        self.ionized_areas_indices = sorted(range(self.n_lightning), key=lambda i: self.lightning[i]['best_score'], reverse=(self.objective == 'max'))[:int(self.n_lightning * 0.2)]
        self.ionized_areas_positions = [self.lightning[i]['position'] for i in self.ionized_areas_indices]
        
    def initialize_lightning(self):                
        if self.ionized_areas_positions:
            # Elegir un índice al azar de ionized_areas_positions
            idx = np.random.randint(len(self.ionized_areas_positions))
            selected_area = self.ionized_areas_positions[idx]
            position = selected_area + (self.storm_power * 0.1) 
            position = np.clip(position, *np.array(self.bounds).T)
            
        else:
            bounds_low, bounds_high = np.array(self.bounds).T
            position = np.random.uniform(bounds_low, bounds_high, size=self.dim)
            position = np.clip(position, *np.array(self.bounds).T)
        
        if tuple(position) in self.cache:
            best_score = self.cache[tuple(position)]
        else:
            best_score = self.function(position)
            self.cache[tuple(position)] = best_score
            self.function_evaluations += 1
            self.function_evaluations_list.append(self.function_evaluations)
        
        return {'position': position, 'best_position': np.copy(position), 'best_score': best_score}
           
    def adjust_field_intensity(self):
        iteration_factor = self.iteration_current / max(self.iterations, 1)
        # Primero calculamos una base de intensidad del campo basada en la iteración actual
        if iteration_factor <= 0.5:
            base_intensity = 3 - (iteration_factor / 0.5 * (2.9))
        elif iteration_factor <= 0.7:
            adjusted_factor = (iteration_factor - 0.5) / 0.2  
            base_intensity = 0.1 - (adjusted_factor * (0.09))
        else:
            adjusted_factor = (iteration_factor - 0.7) / 0.3  
            base_intensity = 0.01 * (1 - adjusted_factor)
        
        if self.field_resistance > 0.25:
            # Aumenta la intensidad si la variabilidad es alta para fomentar la exploración
            self.field_intensity = base_intensity * (1 + (self.field_resistance - 0.5))
            
        else:
            # Disminuye la intensidad para fomentar la explotación si la variabilidad es baja
            self.field_intensity = 3 * base_intensity * self.field_resistance
    
    def calculate_field_resistance(self):        
        positions = np.array([ray['position'] for ray in self.lightning])
        max_diff = np.ptp(positions)  # Peak-to-peak (max - min) difference across all dimensions
        max_diff = max(max_diff, 1e-6)  # Avoid division by zero
        self.field_resistance = np.std(positions) / max_diff    
        
    def calculate_storm_power(self):
        #  Bigger number increases explotaition but reduces the exploration capacity
        self.storm_power = self.field_resistance * self.field_intensity ** np.exp(4 * self.field_resistance)   
          
    def branch_and_propagate(self, idx):
        # Reinitialize lightning if they have been stagnant
        pert_range = np.exp(self.field_resistance) 
        if self.stagnation[idx] > 2:
            self.stagnation[idx] = 0
            return self.initialize_lightning()['position']                                                            

        # Propagate the lightning within the high-energy centers to explore nearby high-quality solutions

        if idx in self.ionized_areas_indices:
            new_position = (self.lightning[idx]['position'] * self.storm_power)           

            # Ensure the position is within the search space boundaries
            new_position = np.clip(new_position, *np.array(self.bounds).T)               

        else:
            # For other lightning, branch them based on the average position of the storm centers
            new_position = np.zeros_like(self.lightning[idx]['position'])
            for center_pos in self.ionized_areas_positions: 
                perturbation = np.random.uniform(-pert_range, pert_range, size=self.dim)  
                new_position += (center_pos + 10 * (perturbation * self.storm_power)) 
            new_position /= len(self.ionized_areas_positions)  # Average updated positions           

            # Ensure the position is within the search space boundaries
            new_position = np.clip(new_position, *np.array(self.bounds).T)       

        return new_position    
                                     
    def select_path(self, idx, adapted_path):       
        if tuple(adapted_path) in self.cache:
            adapted_score = self.cache[tuple(adapted_path)]
        else:
            adapted_score = self.function(adapted_path)
            self.cache[tuple(adapted_path)] = adapted_score
            self.function_evaluations += 1
            self.function_evaluations_list.append(self.function_evaluations)
        
        if (self.objective == 'min' and adapted_score < self.lightning[idx]['best_score']) or \
            (self.objective == 'max' and adapted_score > self.lightning[idx]['best_score']):
            self.lightning[idx]['position'] = adapted_path
            self.lightning[idx]['best_position'] = adapted_path
            self.lightning[idx]['best_score'] = adapted_score
            self.stagnation[idx] = 0
        else:
            self.stagnation[idx] += 1
             
    def update_optimal_strike(self, idx, global_best_position, global_best_score):        
        if (self.objective == 'min' and self.lightning[idx]['best_score'] < global_best_score) or \
        (self.objective == 'max' and self.lightning[idx]['best_score'] > global_best_score):
            global_best_position = self.lightning[idx]['best_position']
            global_best_score = self.lightning[idx]['best_score']
            self.update_best_positions_history(self.lightning[idx]['best_position'], self.lightning[idx]['best_score'])
        return global_best_position, global_best_score
    
    def update_best_positions_history(self, position, score):       
        self.best_positions.append((score, position))
        self.best_positions.sort(key=lambda x: x[0], reverse=(self.objective == 'max'))
        self.best_positions = self.best_positions[:5]  # Keep only the 5 best positions
                   
    def optimize(self):            
            if not callable(self.function):
                raise ValueError("Function must be callable.")
           
            global_best_position = self.lightning[0]['position'].copy()
            global_best_score = self.lightning[0]['best_score']
            
            for iteration in range(self.iterations):
                if self.function_evaluations >= self.max_evaluations:
                    print("Max evaluations reached, stopping optimization.")
                    break  # Stop optimization if maximum evaluations are reached
                self.iteration_current = iteration
                iteration_start_time = time.time()
                self.identify_ionized_areas()
                self.adjust_field_intensity()
                self.calculate_field_resistance()
                self.calculate_storm_power()
                            
                for idx in range(self.n_lightning):
                    adapted_path = self.branch_and_propagate(idx)
                    self.select_path(idx, adapted_path)
                    global_best_position, global_best_score = self.update_optimal_strike(idx, global_best_position, global_best_score)
                                      
                iteration_duration = time.time() - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.objective_values.append(global_best_score)
                self.best_positions.append(global_best_position)

                if self.verbose:
                    def format_global_best(global_best):
                            if isinstance(global_best, np.ndarray):
                                # Es un array, utilizamos np.array2string para convertirlo a string
                                return np.array2string(global_best)
                            else:
                                # Es un número flotante, utilizamos formateo de string directamente
                                return f"{global_best:.20e}"

                        # Dentro de tu código donde imprimes el resultado:
                    print(f"Iteration: {iteration+1}, Field int: {self.field_intensity:.3f}, Field res: {self.field_resistance:.3f}, Current best: {self.lightning[0]['best_score']:.20e}, Global best: {format_global_best(global_best_score)}, Iteration time: {iteration_duration:.5f} seconds, Func. Eval: {self.function_evaluations}")

                                    
            return global_best_position, global_best_score  

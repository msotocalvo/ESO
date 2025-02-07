import numpy as np
import time
from functools import lru_cache

class ESO:
    def __init__(self, function, pop_size=50, max_iter=1000, max_eval=500000, objective='min', bounds=None, verbose=True):
        self.cache = {}        
        self.ionized_areas_index = [] 
        self.ionized_areas_positions = [] 
        self.storm_power = 0 
        self.bounds = bounds or [(0, 1) for _ in range(2)] 
        self.dim = len(self.bounds) 
        self.function = function 
        self.function_evaluations = 0 
        self.function_evaluations_list = []
        self.lightning = np.array([self.initialize_lightning() for _ in range(pop_size)]) 
        self.n_lightning = pop_size 
        self.objective = objective 
        self.iterations = max_iter 
        self.iteration_current = 0 
        self.field_resistance = 0 
        self.field_intensity = 0 
        self.ke = 0    
        self.verbose = verbose 
        self.stagnation = np.zeros(pop_size, dtype=int) 
        self.objective_values = [] 
        self.iteration_times = []  
        self.best_positions = [] 
        self.history = [[] for _ in range(pop_size)]     
        self.max_evaluations = max_eval    
        self.field_intensity_history = []
        self.field_resistance_history = []
        self.field_elasticity_history = []   
        self.storm_power_history = []       
    
    
    @lru_cache(maxsize=None)
    def exp(self, x):       
        try:           
            x = float(x)            
            # Special cases
            if np.isnan(x):
                return np.nan
            if np.isinf(x):
                return 0.0 if x < 0 else np.inf
                
            # Limit the values to avoid overflow
            if x > 709:  # log(np.finfo(np.float64).max)
                return np.finfo(np.float64).max
            if x < -709:
                return np.finfo(np.float64).min
                
            return np.exp(x)
            
        except Exception as e:
            # In case of unexpected error, return a safe value
            print(f"Warning: exp calculation failed with {str(e)}. Using fallback value.")
            return np.finfo(np.float64).eps 

    @lru_cache(maxsize=None)
    def log(self, x):
       
        try:            
            x = float(x)
            
            # Special cases
            if np.isnan(x):
                return np.nan
            if np.isinf(x):
                return np.inf if x > 0 else np.nan
            if x <= 0:
                # For negative values and zero return a very small value          
                return np.log(np.finfo(np.float64).tiny)
                
            # For values very close to zero
            if x < np.finfo(np.float64).tiny:
                return np.log(np.finfo(np.float64).tiny)
                
            return np.log(x)
            
        except Exception as e:
            # In case of unexpected error, return a safe value
            print(f"Warning: log calculation failed with {str(e)}. Using fallback value.")
            return np.log(np.finfo(np.float64).tiny)
    
    def logistic_curve(self, value, multiplier):
        return 1 / (1 + self.exp(-multiplier * (value - np.abs(self.log(1 - value + 10e-50)))))
          
    def identify_ionized_areas(self):
        self.ionized_areas_index = sorted(range(self.n_lightning), key=lambda i: self.lightning[i]['best_score'], 
                                            reverse=(self.objective == 'max'))[:int(self.n_lightning * (self.field_resistance / 2))]
        self.ionized_areas_positions = [self.lightning[i]['position'] for i in self.ionized_areas_index]
        
    def initialize_lightning(self):                
        if self.ionized_areas_positions:
            # Chooses a random index among the ionized areas
            idx = np.random.randint(len(self.ionized_areas_positions))
            selected_area = self.ionized_areas_positions[idx]
            position = selected_area + self.storm_power 
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
       
    def calculate_field_resistance(self):        
        positions = np.array([ray['position'] for ray in self.lightning])
        max_diff = np.ptp(positions)  
        max_diff = max(max_diff, 1e-6)  
        self.field_resistance = (np.std(positions) / max_diff)         
   
    def calculate_ke(self):
        beta = self.logistic_curve(self.field_resistance, self.exp(self.field_resistance) / (self.field_resistance + 10e-50))       
        self.ke = self.exp(self.field_resistance) + (self.exp(1 - self.field_resistance) * np.abs(self.log(self.field_resistance + 10e-50))) * beta        
         
    def adjust_field_intensity(self):     
                    
        iteration_factor = self.iteration_current / max(self.iterations, 1)            
        ganma = 1 / (1 + self.exp(-(self.exp(self.field_resistance) / (self.field_resistance + 10e-50)) * 
                                  (self.field_resistance - np.abs(np.log(1 - iteration_factor)))))
        self.field_intensity = 10e-50 + self.ke * ganma       
        
    def calculate_storm_power(self):
        self.storm_power = self.field_resistance * self.field_intensity ** self.ke   
    
    def branch_and_propagate(self, idx):
        # Reinitialize lightning if they have been stagnant              
        if self.stagnation[idx] > 2:
            self.stagnation[idx] = 0
            return self.initialize_lightning()['position']                                                            

        # Propagate the lightning within the high-energy centers
        if idx in self.ionized_areas_index:
            new_position = (self.lightning[idx]['position'] * self.storm_power)           
            # Ensure the position is within the search space boundaries
            new_position = np.clip(new_position, *np.array(self.bounds).T)               
        else:
            # For other lightning, branch them based on the average position of the storm centers
            if not self.ionized_areas_positions:  # Si no hay áreas ionizadas
                # Generar una nueva posición aleatoria dentro de los límites
                bounds_low, bounds_high = np.array(self.bounds).T
                new_position = np.random.uniform(bounds_low, bounds_high, size=self.dim)
            else:
                accumulated_position = np.zeros_like(self.lightning[idx]['position'])
                num_valid_updates = 0
                
                for center_pos in self.ionized_areas_positions:
                    perturbation = np.random.uniform(-self.ke, self.ke, size=self.dim)
                    update = (center_pos + (perturbation * self.storm_power * np.exp(self.ke)))
                    
                    # Verify update
                    if np.all(np.isfinite(update)):
                        accumulated_position += update
                        num_valid_updates += 1
                
                # Calculate mean if uptade was valid
                if num_valid_updates > 0:
                    new_position = accumulated_position / num_valid_updates
                else:
                    # If update not valid generate ramdon position wihtin bounds
                    bounds_low, bounds_high = np.array(self.bounds).T
                    new_position = np.random.uniform(bounds_low, bounds_high, size=self.dim)

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
        if isinstance(global_best_score, np.ndarray):
            global_best_score = global_best_score.item() 

        if (self.objective == 'min' and self.lightning[idx]['best_score'] < global_best_score) or \
        (self.objective == 'max' and self.lightning[idx]['best_score'] > global_best_score):
            global_best_position = self.lightning[idx]['best_position']
            global_best_score = self.lightning[idx]['best_score']
        return global_best_position, global_best_score
    
    def optimize(self):            
        if not callable(self.function):
            raise ValueError("Function must be callable.")
        
        global_best_position = self.lightning[0]['position'].copy()
        global_best_score = self.lightning[0]['best_score']
                    
        for iteration in range(self.iterations):
            if self.function_evaluations >= self.max_evaluations:
                print("Max evaluations reached, stopping optimization.")
                break 
            self.iteration_current = iteration
            iteration_start_time = time.time()
            self.identify_ionized_areas()
            self.adjust_field_intensity()
            self.calculate_field_resistance()
            self.calculate_ke()
            self.calculate_storm_power()
                                        
            for idx in range(self.n_lightning):
                adapted_path = self.branch_and_propagate(idx)
                self.select_path(idx, adapted_path)
                global_best_position, global_best_score = self.update_optimal_strike(idx, global_best_position, global_best_score)
            
            iteration_duration = time.time() - iteration_start_time
            self.iteration_times.append(iteration_duration)
            self.objective_values.append(global_best_score)
            self.best_positions.append(global_best_position)
            
            self.field_intensity_history.append(self.field_intensity)
            self.field_resistance_history.append(self.field_resistance)
            self.field_elasticity_history.append(self.ke)
            self.storm_power_history.append(self.storm_power)

            if self.verbose:
                def format_global_best(global_best):
                    if isinstance(global_best, np.ndarray):
                        global_best = global_best.item()
                    return f"{global_best:.20e}"
                        
                def format_current_best(current_best):
                    if isinstance(current_best, np.ndarray):
                        return np.array2string(current_best)
                    else:
                        return f"{current_best:.20e}"        

                print(f"Iteration: {iteration+1}, Storm P: {self.storm_power }, Field int: {self.field_intensity:.3f}, Field res: {self.field_resistance:.3f}, ke: {self.ke}, Current best: {format_current_best(self.lightning[0]['best_score'])}, Global best: {format_global_best(global_best_score)}, Iteration time: {iteration_duration:.5f} seconds, Func. Eval: {self.function_evaluations}")
                                
        return global_best_position, global_best_score

# ESO
Electric Storm Optimization (ESO) is an innovative optimization algorithm inspired by the dynamics of natural electric storms. This algorithm combines elements of swarm intelligence and adaptive strategies, making it highly effective in navigating complex, multi-dimensional search spaces with constraints. The core concept involves 'lightnings'—representative of potential solutions—propagating through the search space, guided by an adaptive 'electric field.' This field dynamically adjusts its intensity, ensuring a balanced exploration and exploitation approach.

ESO's adaptability is a standout feature, allowing it to adjust parameters in response to the evolving search landscape. It effectively avoids stagnation, continuously progressing towards optimal solutions by tracking global and local bests and employing crossover points for solution enhancement. Overall, Electric Storm Optimization offers a robust and adaptive approach to complex optimization problems, drawing inspiration from nature to optimize efficiency.

# How to use it:

    from ESO import ESO
    
    # Define your objective function        
    def objective_function(x):
        # Your implementation here
        return computed_value       
    
    # Set up the optimization problem
    optimizer = ESO(
        function = objective_function,
        n_lightning = 50,  # Number of rays (solutions)
        iterations = 10000,  # Number of iterations
        max_eval = 500000,  # Maximum number of objective function evaluations
        objective = 'min',  # 'min' for minimization, 'max' for maximization
        bounds = [(lower_bound, upper_bound), ...],  # Bounds for each dimension
        verbose = True  # Set to True for iteration details
    )
    
    # Perform optimization
    best_position, best_score = optimizer.optimize()
    
    # Output the result
    print(f"Best Position: {best_position}")
    print(f"Best Score: {best_score}")

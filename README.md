# Electrical Storm Optimization (ESO)

A novel metaheuristic optimization algorithm inspired by electrical storm dynamics in nature. ESO combines swarm intelligence with adaptive field mechanics to efficiently explore complex, multi-dimensional search spaces.

## Key Features

- **Dynamic Field Adaptation**: Automatically adjusts field intensity and resistance based on the search landscape
- **Intelligent Branching**: Lightning propagation mechanisms that balance exploration and exploitation
- **Ionized Areas**: Strategic use of promising regions to guide the search process
- **Memory Optimization**: Built-in caching system to avoid redundant function evaluations
- **Robust Error Handling**: Comprehensive management of numerical edge cases
- **Performance Tracking**: Detailed history of metrics and optimization progress

## Quick Start

```python
from eso import ESO

# Define your objective function
def sphere(x):
    return sum(xi**2 for xi in x)

# Configure optimizer
optimizer = ESO(
    function=sphere,          # Objective function to minimize/maximize
    pop_size=50,             # Population size (number of lightning bolts)
    max_iter=1000,           # Maximum iterations
    max_eval=500000,         # Maximum function evaluations
    objective='min',         # 'min' for minimization, 'max' for maximization
    bounds=[(-10, 10)]*2,    # Search space bounds per dimension
    verbose=True             # Enable progress output
)

# Run optimization
best_position, best_score = optimizer.optimize()
```

## Advanced Usage

```python
# Example with custom settings and constraints
optimizer = ESO(
    function=your_function,
    pop_size=100,            # Larger population for complex problems
    max_iter=2000,           # Extended iteration limit
    max_eval=1000000,        # Increased evaluation budget
    objective='max',         # Maximization problem
    bounds=[(0, 1)]*5,       # 5-dimensional problem with [0,1] bounds
    verbose=True
)

# Access optimization history
print(f"Field Intensity History: {optimizer.field_intensity_history}")
print(f"Field Resistance History: {optimizer.field_resistance_history}")
print(f"Storm Power History: {optimizer.storm_power_history}")
```

## Algorithm Parameters

- `function`: Target objective function to optimize
- `pop_size`: Number of lightning bolts in the population
- `max_iter`: Maximum number of iterations
- `max_eval`: Maximum number of function evaluations
- `objective`: Optimization direction ('min' or 'max')
- `bounds`: List of tuples defining the search space boundaries
- `verbose`: Enable/disable progress output

## Performance Monitoring

The algorithm tracks various metrics during optimization:
- Function evaluation history
- Field intensity and resistance evolution
- Storm power dynamics
- Best solution progression
- Iteration timing statistics

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use ESO in your research, please cite:

```bibtex
@article{eso2024,
    title={Electrical Storm Optimization (ESO) Algorithm:Theoretical foundations, analysis, and application to engineering problems},
    author={[Soto Calvo M.; Lee Han S.]},
    journal={[Journal Name]},
    doi={[DOI]},
    year={2025}
}
```
## Disclaimers
- The Matlab implementation of the ESO is still under development and its performance is not yet ensured. 


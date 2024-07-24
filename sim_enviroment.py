######## Import libraries
import numpy as np
import os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import datetime as datetime
from ESO import ESO
from mealpy import HS, TS,GA,DE,SA,PSO,ABC,ACOR,GWO,WOA,CSA,EFO,MFO,BFO,FFA,BA, BBO, CRO,FPA, HHO, SHADE ,FloatVar
import FunctionUtil
from opfunu.cec_based import cec2022, cec2021
import opfunu
import baycomp
import scikit_posthocs as sp
from CEC2021_RWCMO import *
from BMF import *
import concurrent.futures  
from autorank import autorank
###### Simulation parameters ############################################################
num_simulations = 50 # Number of simulations
pop_size = 50 # Size of the population
max_iter = 1000 # Max number of iterations
max_eval = 50_000 # Max number of function evaluation
D = 100 
problem_group = 'All' # Group of functions to be optimized 

##### General Benchmark Functions
test_functions = [
    {'func': Booth.function, 'name': 'F1', 'optimal': Booth.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': Cone.function, 'name': 'F2', 'optimal': Cone.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': ChungReynolds.function, 'name': 'F3', 'optimal': ChungReynolds.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Ellipse.function, 'name': 'F4', 'optimal': Ellipse.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': ElAttarVidyasagarDutta.function, 'name': 'F5', 'optimal': ElAttarVidyasagarDutta.optimal, 'bounds': [ElAttarVidyasagarDutta.bounds for _ in range(D)]},
    {'func': Leon.function, 'name': 'F6', 'optimal': Leon.optimal, 'bounds': [(-1.2, 1.2) for _ in range(D)]},
    {'func': PowellSingular1.function, 'name': 'F7', 'optimal': PowellSingular1.optimal, 'bounds': [(-4, 5) for _ in range(D)]},
    {'func': PowellSingular2.function, 'name': 'F8', 'optimal': PowellSingular2.optimal, 'bounds': [(-4, 5) for _ in range(D)]},
    {'func': PowellSum.function, 'name': 'F9', 'optimal': PowellSum.optimal, 'bounds': [(-1, 1) for _ in range(D)]},
    {'func': Ridge.function, 'name': 'F10', 'optimal': Ridge.optimal, 'bounds': [Ridge.bounds for _ in range(D)]},
    {'func': Rosenbrock.function, 'name': 'F11', 'optimal': Rosenbrock.optimal, 'bounds': [(0, 1) for _ in range(D)]},
    {'func': Schwefel01.function, 'name': 'F12', 'optimal': Schwefel01.optimal, 'bounds': [(-500, 500) for _ in range(D)]},
    {'func': Schwefel02.function, 'name': 'F13', 'optimal': Schwefel02.optimal, 'bounds': [(-1, 4) for _ in range(D)]},
    {'func': Schwefel20.function, 'name': 'F14', 'optimal': Schwefel20.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Schwefel21.function, 'name': 'F15', 'optimal': Schwefel21.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Schwefel22.function, 'name': 'F16', 'optimal': Schwefel22.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Schwefel23.function, 'name': 'F17', 'optimal': Schwefel23.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Sphere.function, 'name': 'F18', 'optimal': Sphere.optimal, 'bounds': [(-5.12, 5.12) for _ in range(D)]},
    {'func': Step3.function, 'name': 'F19', 'optimal': Step3.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Stepint.function, 'name': 'F20', 'optimal': Stepint.optimal, 'bounds': [Stepint.bounds for _ in range(5)]},
    {'func': StretchedVSineWave.function, 'name': 'F21', 'optimal': StretchedVSineWave.optimal, 'bounds': [(-5, 5) for _ in range(D)]},
    {'func': SumSquares.function, 'name': 'F22', 'optimal': SumSquares.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': WayburnSeader1.function, 'name': 'F23', 'optimal': WayburnSeader1.optimal, 'bounds': [(-5, 5) for _ in range(D)]},
    {'func': WayburnSeader2.function, 'name': 'F24', 'optimal': WayburnSeader2.optimal, 'bounds': [(-500, 500) for _ in range(D)]},
    {'func': Zirilli.function, 'name': 'F25', 'optimal': Zirilli.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': ackley1.function, 'name': 'F26', 'optimal': ackley1.optimal, 'bounds': [(-32.768, 32.768) for _ in range(D)]},
    {'func': alpine_1.function, 'name': 'F27', 'optimal': alpine_1.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': BentCigar.function, 'name': 'F28', 'optimal': BentCigar.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Corana.function, 'name': 'F29', 'optimal': Corana.optimal, 'bounds': [(-500,500) for _ in range(D)]},
    {'func': CrownedCross.function, 'name': 'F30', 'optimal': CrownedCross.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': CrossLegTable.function, 'name': 'F31', 'optimal': CrossLegTable.optimal, 'bounds': [(-10, 10) for _ in range(D)]},
    {'func': Csendes.function, 'name': 'F32', 'optimal': Csendes.optimal, 'bounds': [(-1, 1) for _ in range(D)]},
    {'func': Damavandi.function, 'name': 'F33', 'optimal': Damavandi.optimal, 'bounds': [(0, 14) for _ in range(D)]},
    {'func': Dolan.function, 'name': 'F34', 'optimal': Dolan.optimal, 'bounds': [Dolan.bounds for _ in range(D)]},
    {'func': DropWave.function, 'name': 'F35', 'optimal': DropWave.optimal, 'bounds': [DropWave.bounds for _ in range(D)]},
    {'func': EggCrate.function, 'name': 'F36', 'optimal': EggCrate.optimal, 'bounds': [(-5, 5) for _ in range(D)]},
    {'func': Griewank.function, 'name': 'F37', 'optimal': Griewank.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': OddSquare.function, 'name': 'F38', 'optimal': OddSquare.optimal, 'bounds': [OddSquare.bounds] * 19},
    {'func': Price2.function, 'name': 'F39', 'optimal': Price2.optimal, 'bounds': [Price2.bounds for _ in range(D)]},
    {'func': Rastrigin.function, 'name': 'F40', 'optimal': Rastrigin.optimal, 'bounds': [(-5.12, 5.12) for _ in range(D)]},
    {'func': RosenbrockModified.function, 'name': 'F41', 'optimal': RosenbrockModified.optimal, 'bounds': [(-2, 2) for _ in range(D)]},
    {'func': Salomon.function, 'name': 'F42', 'optimal': Salomon.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Schaffer.function, 'name': 'F43', 'optimal': Schaffer.optimal, 'bounds': [(-100, 100) for _ in range(D)]},
    {'func': Stochastic.function, 'name': 'F44', 'optimal': Stochastic.optimal, 'bounds': [Stochastic.bounds for _ in range(2)]},
    {'func': Weierstrass.function, 'name': 'F45', 'optimal': Weierstrass.optimal, 'bounds': [(-0.5, 0.5) for _ in range(D)]},
    {'func': XinSheYang1.function, 'name': 'F46', 'optimal': XinSheYang1.optimal, 'bounds': [(-5, 5) for _ in range(D)]},
    {'func': XinSheYang2.function, 'name': 'F47', 'optimal': XinSheYang2.optimal, 'bounds': [(-2*np.pi, 2*np.pi) for _ in range(D)]},
    {'func': XinSheYang3.function, 'name': 'F48', 'optimal': XinSheYang3.optimal, 'bounds': [(-20, 20) for _ in range(D)]},
    {'func': ZeroSum.function, 'name': 'F49', 'optimal': ZeroSum.optimal, 'bounds': [ZeroSum.bounds for _ in range(D)]},
    {'func': Zimmerman.function, 'name': 'F50', 'optimal': Zimmerman.optimal, 'bounds': [Zimmerman.bounds for _ in range(D)]},
    {'func': cec2022.F12022(10).evaluate, 'name': 'F51-D10', 'optimal': 300, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F22022(10).evaluate, 'name': 'F52-D10', 'optimal': 400, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F32022(10).evaluate, 'name': 'F53-D10', 'optimal': 600, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F42022(10).evaluate, 'name': 'F54-D10', 'optimal': 800, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F52022(10).evaluate, 'name': 'F55-D10', 'optimal': 900, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F62022(10).evaluate, 'name': 'F56-D10', 'optimal': 1800, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F72022(10).evaluate, 'name': 'F57-D10', 'optimal': 2000, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F82022(10).evaluate, 'name': 'F58-D10', 'optimal': 2200, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F92022(10).evaluate, 'name': 'F59-D10', 'optimal': 2300, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F102022(10).evaluate, 'name': 'F60-D10', 'optimal': 2400, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F112022(10).evaluate, 'name': 'F61-D10', 'optimal': 2600, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F122022(10).evaluate, 'name': 'F62-D10', 'optimal': 2700, 'bounds': [(-100, 100) for _ in range(10)]},
    {'func': cec2022.F12022(20).evaluate, 'name': 'F51', 'optimal': 300, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F22022(20).evaluate, 'name': 'F52', 'optimal': 400, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F32022(20).evaluate, 'name': 'F53', 'optimal': 600, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F42022(20).evaluate, 'name': 'F54', 'optimal': 800, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F52022(20).evaluate, 'name': 'F55', 'optimal': 900, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F62022(20).evaluate, 'name': 'F56', 'optimal': 1800, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F72022(20).evaluate, 'name': 'F57', 'optimal': 2000, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F82022(20).evaluate, 'name': 'F58', 'optimal': 2200, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F92022(20).evaluate, 'name': 'F59', 'optimal': 2300, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F102022(20).evaluate, 'name': 'F60', 'optimal': 2400, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F112022(20).evaluate, 'name': 'F61', 'optimal': 2600, 'bounds': [(-100, 100) for _ in range(20)]},
    {'func': cec2022.F122022(20).evaluate, 'name': 'F62', 'optimal': 2700, 'bounds': [(-100, 100) for _ in range(20)]},     
    
]   

def create_algorithm(name, func, bounds, **params):
    if name == 'ABC':
        return ABC.OriginalABC(**params)
    elif name == 'ACO':
        return ACOR.OriginalACOR(**params)
    elif name == 'DE':
        return DE.OriginalDE(**params)
    elif name == 'GA':
        return GA.BaseGA(**params)
    elif name == 'GWO':
        return GWO.OriginalGWO(**params)
    elif name == 'HHO':
        return HHO.OriginalHHO(**params)
    elif name == 'MFO':
        return MFO.OriginalMFO(**params)
    elif name == 'PSO':
        return PSO.OriginalPSO(**params)
    elif name == 'WOA':
        return WOA.OriginalWOA(**params)
    elif name == 'LSHADE':
        return SHADE.L_SHADE(**params)    
    elif name == 'ESO':
        return ESO(function=func, bounds=bounds, **params)
    if name == 'BFO':
        return BFO.ABFO(**params) 
    elif name == 'BA':
        return BA.OriginalBA(**params)
    elif name == 'BBO':
        return BBO.OriginalBBO(**params)
    elif name == 'CRO':
        return CRO.OriginalCRO(**params)   
    elif name == 'FPA':
        return FPA.OriginalFPA(**params)
    elif name == 'FFA':
        return FFA.OriginalFFA(**params)
    elif name == 'CS':
        return CSA.OriginalCSA(**params)
    elif name == 'HS':
        return HS.OriginalHS(**params)
    elif name == 'SA':
        return SA.OriginalSA(**params)
    elif name == 'TS':
        return TS.OriginalTS(**params)

def run_algorithm(test, name):
    func = test['func']
    bounds = test['bounds']
    alg_params = {
    'epoch': max_iter if name != 'ESO' else None,
    'pop_size': pop_size,
    'n_limits': 50 if name == 'ABC' else None,
    'sample_count': 25 if name in ['ACO', 'CS'] else None,
    'intent_factor': 0.5 if name in ['ACO', 'CS'] else None,
    'zeta': 1.0 if name in ['ACO', 'CS'] else None,
    'wf': 0.7 if name == 'DE' else None,
    'cr': 0.9 if name == 'DE' else None,
    'strategy': 0 if name == 'DE' else None,
    'pc': 0.9 if name == 'GA' else None,
    'pm': 0.05 if name == 'GA' else None,
    'c1': 2.05 if name == 'PSO' else None,
    'c2': 2.05 if name == 'PSO' else None,
    'w': 0.4 if name == 'PSO' else None,
    'miu_f': 0.5 if name in ['LSHADE'] else None,
    'miu_cr': 0.5 if name in ['LSHADE'] else None,
    'max_iter': max_iter if name == 'ESO' else None,
    'max_eval': max_eval if name == 'ESO' else None,
    'verbose': False if name == 'ESO' else None,
    'C_s': 0.1 if name == 'BFO' else None,
    'C_e': 0.001 if name == 'BFO' else None,
    'Ped': 0.01 if name == 'BFO' else None,
    'Ns': 4 if name == 'BFO' else None,
    'N_adapt': 2 if name == 'BFO' else None,
    'N_slplit': 40 if name == 'BFO' else None,
    'loudness': 0.8 if name == 'BA' else None,
    'pulse_rate': 0.95 if name == 'BA' else None,
    'pf_min': 0.1 if name == 'BA' else None,
    'pf_max': 10.0 if name == 'BA' else None,
    'p_m': 0.01 if name == 'BBO' else None,
    'n_elites': 2 if name == 'BBO' else None,
    'po': 0.4 if name == 'CRO' else None,
    'Fb': 0.9 if name == 'CRO' else None,
    'Fa': 0.1 if name == 'CRO' else None,
    'Fd': 0.1 if name == 'CRO' else None,
    'Pd': 0.5 if name == 'CRO' else None,
    'GCR': 0.1 if name == 'CRO' else None,
    'gamma_min': 0.02 if name == 'CRO' else None,
    'gamma_max': 0.2 if name == 'CRO' else None,
    'n_trials': 5 if name == 'CRO' else None,   
    'p_s': 0.8 if name == 'FPA' else None,
    'levy_multiplier': 0.2 if name == 'FPA' else None,
    'gamma': 0.001 if name == 'FFA' else None,
    'beta_base': 2 if name == 'FFA' else None,
    'alpha': 0.2 if name == 'FFA' else None,
    'alpha_damp': 0.99 if name == 'FFA' else None,
    'delta': 0.05 if name == 'FFA' else None,
    'exponent': 2 if name == 'FFA' else None,
    'sample_count': 25 if name == 'CS' else None,
    'intent_factor': 0.5 if name == 'CS' else None,
    'zeta': 1.0 if name == 'CS' else None,
    'tabu_size': 5 if name == 'TS' else None,
    'neighbour_size': 10 if name == 'TS' else None,
    'perturbation_scale': 0.05 if name == 'TS' else None,
    'temp_init': 100 if name == 'SA' else None,
    'step_size': 0.1 if name == 'SA' else None
}
    
    alg_func = create_algorithm(name, func, bounds, **{k: v for k, v in alg_params.items() if v is not None})
    problem = {
        "obj_func": func,
        "bounds": FloatVar(lb=[b[0] for b in bounds], ub=[b[1] for b in bounds]),
        "minmax": "min",
        "log_to": None,        
    }
    term_dict = {
        "max_fe": max_eval,
        "max_epoch": max_iter        
    }
    start_time = time.time()    
    
    result = alg_func.solve(problem=problem, termination=term_dict) if name != 'ESO' else alg_func.optimize()
    alg_time = time.time() - start_time
    return {
        'alg_name': name,
        'best_score': result.target.fitness if name != 'ESO' else result[1],
        'execution_time': alg_time,
        'best_solution': result.solution if name != 'ESO' else result[0],
    }
    
if __name__ == '__main__':
    # List to store algorithm's names
    algorithm_names = [  'ESO', 'ABC','ACO', 'DE', 'GA', 'GWO', 'HHO', 'MFO', 'PSO', 'WOA', 'LSHADE', 'BFO', 'BA', 'BBO', 'CRO', 'FPA', 'FFA', 'CS',  'HS',  'SA', 'TS']
    

    # stores the results from the algorithms 
    history = {
        'Algorithm': [],
        'Function': [],
        'Best Score': [],
        'Accuracy': [],
        'Execution Time': [],
        'Best Solution': [],
                 
    }

    # Execute the algorithm in a parallel enviroment 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in range(num_simulations):
            for test in test_functions:
                func_name = test['name']  
                known_optimum = test['optimal']
                futures = {executor.submit(run_algorithm, test, name): name for name in algorithm_names}
                for future in concurrent.futures.as_completed(futures):
                    data = future.result()
                    distance = np.linalg.norm([known_optimum - data['best_score']])
                    accuracy = np.abs(1 /(1 + distance)) 
                    alg_name = futures[future] 

                    # Save the results
                    history['Algorithm'].append(alg_name)
                    history['Function'].append(func_name)
                    history['Best Score'].append(data['best_score'])                    
                    history['Accuracy'].append(accuracy)
                    history['Execution Time'].append(data['execution_time'])
                    history['Best Solution'].append(str(data['best_solution']))  # Convertir soluciones a string para evitar problemas de formato                    

                    # Print the progress 
                    remaining_runs = len(test_functions) * num_simulations - (len(history['Algorithm']) // len(algorithm_names))
                    progress = ( 1 - (remaining_runs / (len(test_functions) * num_simulations))) * 100
                    print(f"Completed: {alg_name} for {func_name}. Remaining: {remaining_runs}. Progress: {progress:.3f} %")
                    

    df = pd.DataFrame(history)
    # with pd.ExcelWriter('history.xlsx', engine='openpyxl') as writer:
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('Results', exist_ok=True)
    filename = f"Results/{problem_group}_{current_datetime}_history_raw.xlsx"
    df.to_excel(filename, index=False)
    

    success_threshold = 10E-8


    performance_metrics = []
    bayesian_results = []

    for test in test_functions:
        func_name = test['name']
        known_optimum = test['optimal']       

        for algo in algorithm_names:
            scores = [history['Best Score'][i] for i in range(len(history['Algorithm'])) if history['Algorithm'][i] == algo and history['Function'][i] == func_name]
            times = [history['Execution Time'][i] for i in range(len(history['Algorithm'])) if history['Algorithm'][i] == algo and history['Function'][i] == func_name]

            if scores:
                best_score = min(scores, key=lambda x: abs(x - known_optimum))
                worst_score = max(scores, key=lambda x: abs(x - known_optimum))
                std_dev = np.std(scores, ddof=1)
                avg_score = np.mean(scores)
                distance = np.linalg.norm([known_optimum - avg_score])
                num_successes = sum(abs(score - known_optimum) <= success_threshold for score in scores)
                success_ratio = num_successes / len(scores) if scores else 0
                avg_time = np.mean(times) if times else 0
                accuracy = (1 /(1 + distance))                         

                # Append results to data_list
                performance_metrics.append({
                    'Function': func_name,
                    'Algorithm': algo,
                    'Best Score': best_score,
                    'Worst Score': worst_score,
                    'Average Score': avg_score,
                    'Standard Deviation': std_dev,
                    'Euclidean Distance': distance,
                    'Accuracy': accuracy,
                    'Success Ratio': success_ratio,
                    'Average Time': avg_time,                    
                })  
                

    df_metrics = pd.DataFrame(performance_metrics)   
      
    # Save the results 
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_metrics = f"Results/{problem_group}_{current_datetime}_performance_metrics.xlsx"
    df_metrics.to_excel(filename_metrics, index=False)  
        
    if len(test_functions) >= 5:
        # Pivot the data so that the classifiers become columns and datasets are the rows    
        pivot_data = df_metrics.pivot(index='Function', columns='Algorithm', values='Accuracy')

        # Apply the Bayesian signed-rank test using autorank
        result_bayesian = autorank(pivot_data, alpha=0.05, verbose=False, approach='bayesian')    
    else:
        print(f"Not enough samples to run Bayesian signed-rank test")
                                       
   
       
    
    
   
    
   

    
    
    

    

######## Import libraries
import numpy as np
import os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import datetime as datetime
# from ESO import ESO
from ESO import ESO
from mealpy import HS, TS,GA,DE,SA,PSO,ABC,ACOR,GWO,WOA,CSA,EFO,GOA,BFO,FFA,BA, FA, BBO, CRO, EP, NRO, FPA, HHO ,FloatVar
import FunctionUtil
from opfunu.cec_based import cec2022, cec2021, cec2020
from visualize import plot_func
from OddSquare import OddSquare

###### Simulation parameters ############################################################
num_simulations = 4 # Number of simulations
pop_size = 50 # Size of the population
max_iter = 10000 # Max number of iterations
D = 50 # Dimension equal D * 2 (Beacause functions are expresed already with 2D boundaries)
visualize = False  # Whether to visualize the function to be optimized
entire_set = True # Whether to use the entire set of algorithms or a reduced one
problem_group = 'CEC2022' # Group of functions to be optimized 

##### General Benchmark Functions
def p(x):
    return 1 if x >= 0 else 0
a = 5  # Example value for 'a'
K = 5  # Example value for 'K', often it is 5*n^2


odd_square_instance = OddSquare(dimensions=10)
####### Continuous, Differentiable, Multimodal (10 functions)
if problem_group == 'G1':
    test_functions = [  
                           
        # {'func': lambda x: -20 * np.exp(-0.2 * np.sqrt(sum(x_i**2 for x_i in x) / len(x))) - np.exp(sum(np.cos(2 * np.pi * x_i) for x_i in x) / len(x)) + 20 + np.e,'name': 'Ackley 1',  'optimal': 0.00000, 'bounds': [(-32.768, 32.768), (-32.768, 32.768)] * D}, # OK            
        # {'func': lambda x:np.cos(x[0]) * np.sin(x[1]) - x[0] / (x[1]**2 + 1),'name': 'Adjman', 'optimal': -2.0218,'bounds': [(-1, 2), (-1, 2)] * D}, # OK            
        # {'func': lambda x:sum([xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x]),'name': 'Griewank', 'optimal': 0.00000,'bounds': [(-600, 600), (-600, 600)] * D},
        # {'func': lambda x:sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x),'name': 'Rastrigin', 'optimal': 0.00000,'bounds': [(-5.12, 5.12), (-5.12, 5.12)] * D}, # OK 
        {'func': lambda x: np.sum(np.square(x)), 'name': 'Sphere', 'optimal': 0,'bounds': [(-5.12, 5.12), (-5.12, 5.12)] * D}, # OK
        # {'func': lambda x: -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))),'name': 'Holder Table', 'optimal': -19.208,'bounds': [(-10, 10), (-10, 10)] * D}, # OK
        # {'func': lambda x: np.exp(np.sin(50 * x[0])) + np.sin(60 * np.exp(x[1])) + np.sin(70 * np.sin(x[0])) + np.sin(np.sin(80 * x[1])) - np.sin(10 * (x[0] + x[1])) + (x[0]**2 + x[1]**2) / 4,'name': 'Trefethen','optimal': -3.3068,'bounds': [(-10, 10), (-10, 10)] * D}, # OK
        # {'func': lambda x: -np.sin(2.2*np.pi*x[0] + 0.5) * ((2 - abs(x[0])) / 2)**3 - np.sin(0.5*np.pi*x[1]**2 + 0.5) * ((2 - abs(x[1])) / 2)**3,'name': 'Ursem 3','optimal': -1.285,'bounds': [(-2, 2), (-1.5, 1.5)] * D}, #OK
        # {'func': lambda x:np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * (xi + 0.5)) for i in range(20)]) for xi in x]) - len(x) * np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * 0.5) for i in range(20)])]),'name': 'Weierstrass', 'optimal': 0.00000,'bounds': [(-0.5, 0.5), (-0.5, 0.5)] * D}, # OK
        # {'func': lambda x: np.exp(-sum([(xi/15)**(2*3) for xi in x])) - 2*np.exp(-sum([xi**2 for xi in x])) * np.prod([np.cos(xi)**2 for xi in x]),'name': 'Xin-She Yang 3','optimal': -1,'bounds': [(-20, 20), (-20, 20)]}, ## GWO fails
        ]

    
####### Continuous, Differentiable, Unimodal (10 functions)
elif problem_group == 'G2':
    test_functions = [
        # {'func': lambda x: -200 * np.exp(-0.02 * np.sqrt((x[0]**2 + x[1]**2)/2)),'name': 'Ackley 2', 'optimal': -200.00, 'bounds': [(-32, 32), (-32, 32)] * D}, # OK
        # {'func': lambda x: sum((x[4*i-4] + 10*x[4*i-3])**2 + 5*(x[4*i-2] - x[4*i-1])**2 + (x[4*i-3] - 2*x[4*i-2])**4 + 10*(x[4*i-4] - x[4*i-1])**4 for i in range(1, len(x)//4 + 1)),'name': 'Powell Singular','optimal': 0.00000,'bounds': [(-4, 5), (-4, 5)] * D}, # OK
        # {'func': lambda x: sum(abs(x_i)**(i+1) for i, x_i in enumerate(x)),'name': 'Powell Sum', 'optimal': 0.00000,'bounds': [(-1, 1), (-1, 1)] *D }, # ESO queda tercero
        # {'func': lambda x: 7*x[0]**2 - 6*np.sqrt(3)*x[0]*x[1] + 13*x[1]**2,'name': 'Rotated Ellipse','optimal':0.0,'bounds': [(-500, 500), (-500, 500)] *D }, # ESO queda septimo
        # {'func': lambda x: 0.5 + (np.sin(np.sqrt(x[0]**2 + x[1]**2))**2 - 0.5) / (1 + 0.001*(x[0]**2 + x[1]**2))**2,'name': 'Scahffer', 'optimal': 0.00000,'bounds': [(-100, 100), (-100, 100)] * D}, # OK
        # {'func': lambda x:0.5 + ((np.sin(x[0]**2 - x[1]**2)**2 - 0.5)/(1 + 0.001*(x[0]**2 + x[1]**2))**2),'name': 'Scahffer 2', 'optimal': 0.00000,'bounds': [(-100, 100), (-100, 100)] * D}, # OK
        # {'func': lambda x: sum(x_i**2 for x_i in x)**2,'name': 'Schwefel', 'optimal': 0.00000,'bounds': [(-500, 500), (-500, 500)] * D}, # OK
        # {'func': lambda x:np.sum(x**2)**2,'name': 'Chung Reynolds', 'optimal': 0.00000,'bounds': [(-5.12, 5.12), (-5.12, 5.12)] * D}, # OK
        # {'func': lambda x:(x[0] - 1)**2 + sum([(i+1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x))]),'name': 'Rosembrock', 'optimal': 0.0,'bounds': [(0, 1) for _ in range(D)]}, # OK
        {'func': lambda x:(1.0 - (np.abs(np.sin(np.pi * (x[0] - 2.0)) * np.sin(np.pi * (x[1] - 2.0)) / (np.pi ** 2) * (x[0] - 2.0) * (x[1] - 2.0) + 1e-8)) ** 5.0 * 2 + (x[0] - 7.0) ** 2.0 + 2 * (x[1] - 7.0) ** 2.0),'name': 'Damavandi', 'optimal': 0.0,'bounds': [(0, 14) for _ in range(D)]}, # OK
               
        # {'func': lambda x: sum(((x[i]**2 + x[i+1]**2)**0.25) * (np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2 + 0.1) for i in range(len(x) - 1)),'name': 'Stretched V Sine Wave', 'optimal': 0, 'bounds': [(-5, 5), (-5, 5)] * D}, # OK 
        # {'func': lambda x: (x[0]**2 + x[1]**2 - 2*x[1])**2 + 0.25*x[1],'name': 'Zettl','optimal': -0.003791,'bounds': [(-5, 10), (-5, 10)] * D}, # OK
    ]

####### Continuous, Non-Differentiable (10 functions)
elif problem_group == 'G3':
    test_functions = [
        {'func': lambda x:(np.sum(x) / len(x) - np.prod(x) ** (1.0 / len(x))) ** 2, 'name': 'Aritmetric Mean', 'optimal': 0, 'bounds': [(0,10),(0, 10)] * D},
        {'func': lambda x:sum([np.abs(xi * np.sin(xi) + 0.1 * xi) for xi in x]), 'name': 'Alpine 1', 'optimal': 0.00000,'bounds': [(-10, 10), (-10, 10)] * D}, # OK
        {'func': lambda x: abs(x[0]**2 + x[1]**2 + x[0] * x[1]) + abs(np.sin(x[0])) + abs(np.cos(x[1])),'name': 'Bartels Conn', 'optimal': 1.00000,'bounds': [(-500, 500), (-500, 500)] * D}, # OK 
        {'func': lambda x:(1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2, 'name':'Beale', 'optimal': 0.00000, 'bounds': [(-4.5,4.5), (-4.5, 4.5)] * D}, # OK 
        {'func': lambda x: 100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0) + 0.01 * (x[0] + 10.0) ** 2.0,'name': 'Bukin 4', 'optimal':  -124.75, 'bounds': [(-15, -5), (-3, 3)] * D}, # La funcion esta mal
        {'func': lambda x: 100 * np.sqrt(abs(x[1] - 0.01 * (x[0]**2))) + 0.01 * abs(x[0] + 10),'name': 'Bukin 6', 'optimal': 0.00000,'bounds': [(-15, -5), (-3, 3)] * D}, # OK
        {'func': lambda x:-0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1)**0.1,'name': 'CrossInTray', 'optimal': -2.0626,'bounds': [(-10, 10), (-10, 10)] * D}, # OK
        {'func': lambda x: (np.abs(x[0] - 5)**2 + np.abs(x[1] - 5)**2),'name': 'Price 1', 'optimal': 0, 'bounds': [(-500, 500), (-500, 500)] * D},
        {'func': lambda x: np.sum(np.abs(x)),'name': 'Schwefel 2.20', 'optimal': 0.0, 'bounds': [(-100, 100) for _ in range(D)]},
        {'func': lambda x: np.max(np.abs(x)), 'name': 'Schwefel 2.21', 'optimal': 0, 'bounds': [(-100,100), (-100, 100)] * D},
        {'func': lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x)),'name': 'Schwefel 2.22', 'optimal': 0.0, 'bounds': [(-100, 100) for _ in range(D)]},
                 
    ]

####### Discontinuous, Non-Differentiable  (7 functions)
elif problem_group == 'G4':
    test_functions = [
    #    {'func': lambda x: odd_square_instance.evaluator(x), 'name': 'Odd Square','optimal': -1.00846728102, 'bounds': [(-5/np.pi,5/np.pi), (-5/np.pi, 5/np.pi)] * D }, 
        {'func': lambda x: sum(0.15 * (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]) - 0.05 * np.sign(x[i]))**2 * [1, 1000, 10, 100][i] if abs(x[i] - (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]))) < 0.05 else [1, 1000, 10, 100][i] * x[i]**2 for i in range(4)),'name': 'Corana', 'optimal': 0.00000,'bounds': [(-500, 500), (-500, 500)] * D }, # OK   
        {'func': lambda x: -0.1 * sum([np.cos(5*np.pi*xi) for xi in x]) - sum([xi**2 for xi in x]),'name': 'Cosine Mixture', 'optimal': -0.9 * D*2,'bounds': [(-1, 1),(-1, 1)] * D }, # OK   
        {'func': lambda x: sum([np.floor(abs(xi)) for xi in x]),'name': 'Step 1', 'optimal': 0.00000,'bounds': [(-100, 100), (-100, 100)] * D}, # OK
        {'func': lambda x:np.prod(np.sum(np.round(2 ** np.atleast_2d(np.arange(1, len(x) + 1)).T * x) * (2. ** (-np.atleast_2d(np.arange(1, len(x) + 1)).T)), axis=0) * (np.arange(0., len(x) * 1.) + 1) + 1),'name': 'Katsuura', 'optimal': 1.0, 'bounds': [(0,100), (0,100)] * D}, # Da error
        {'func': lambda x: np.sum(np.floor(x + 0.5)**2),'name': 'Step 2', 'optimal': 0.00000,'bounds': [(-100, 100), (-100, 100)] * D}, # OK
        {'func': lambda x: p(x[1])*(1 + p(x[0])) + abs(x[0] + 50*p(x[1])*(1 - 2*p(x[0]))) + abs(x[1] + 50*(1 - 2*p(x[1]))), 'name': 'Tripod', 'optimal': 0.00000,'bounds': [(-100, 100), (-100, 100)] * D}, # OK
    ]
        
####### Fixed-dimension multimodal (15 functions) ### Comprobar optimos
elif problem_group == 'G5':
    D = 30
    test_functions = [
        # {'func': lambda x: FunctionUtil.hho_f14(x),'name': 'hho_f14','optimal': 0.0,'bounds':[(-65,65),(-65,65)]},
        # {'func': lambda x: FunctionUtil.hho_f15(x),'name': 'hho_f15','optimal': 0.0,'bounds':[(-5,5), (-5,5),(-5,5), (-5,5)]}, # Pincha bien
        # {'func': lambda x: FunctionUtil.hho_f16(x),'name': 'hho_f16','optimal': 0.0,'bounds':[(-5,5), (-5,5)]},
        # {'func': lambda x: FunctionUtil.hho_f17(x),'name': 'hho_f17','optimal': 0.0,'bounds':[(-5,5), (-5,5)]}, # Pincha bien
        # {'func': lambda x: FunctionUtil.hho_f18(x),'name': 'hho_f18','optimal': 3.0,'bounds':[(-2,2),(-2,2)]},
        # {'func': lambda x: FunctionUtil.hho_f19(x),'name': 'hho_f19','optimal': 0.0,'bounds':[(1,3),(1,3),(1,3)]},
        # {'func': lambda x: FunctionUtil.hho_f20(x),'name': 'hho_f20','optimal': 0.0,'bounds':[(0,1),(0,1),(0,1), (0,1),(0,1), (0,1)]},
        # {'func': lambda x: FunctionUtil.hho_f21(x),'name': 'hho_f21','optimal': 0.0,'bounds':[(0,1), (0,1),(0,1), (0,1)]},
        # {'func': lambda x: FunctionUtil.hho_f22(x),'name': 'hho_f22','optimal': 0.0,'bounds':[(0,1), (0,1),(0,1), (0,1)]},
        # {'func': lambda x: FunctionUtil.hho_f23(x),'name': 'hho_f23','optimal': 0.0,'bounds':[(0,1), (0,1),(0,1), (0,1)]},
        # {'func': lambda x: FunctionUtil.C16(x),'name': 'hho_f24','optimal': 0.0,'bounds':[(-5,5)] * D},
        # {'func': lambda x: FunctionUtil.C18(x),'name': 'hho_f25','optimal': 0.0,'bounds':[(-5,5)] * D},
        # {'func': lambda x: FunctionUtil.C20(x),'name': 'hho_f27','optimal': 0.0,'bounds':[(-5,5)] * D},
        # {'func': lambda x: FunctionUtil.C21(x),'name': 'hho_f28','optimal': 0.0,'bounds':[(-5,5)] * D},
        # {'func': lambda x: FunctionUtil.C25(x),'name': 'hho_f29','optimal': 0.0,'bounds':[(-5,5)] * D},
    ]
    

############################################### CEC Test Suit
elif problem_group == 'CEC2022':
    D = 10
    ########### CEC 2022 (12  functions)
    test_functions = [
        {'func': lambda x: cec2022.F12022(D).evaluate(x), 'name': 'CEC2022 F1', 'optimal': 300, 'bounds': [(-100, 100)] * D}, # ESO is the best
        {'func': lambda x:cec2022.F22022(D).evaluate(x), 'name': 'CEC2022 F2', 'optimal': 400, 'bounds': [(-100, 100)] * D}, # Le da duro
        {'func': lambda x: cec2022.F32022(D).evaluate(x), 'name': 'CEC2022 F3', 'optimal': 600, 'bounds': [(-100, 100)] * D}, # All works well
        {'func': lambda x: cec2022.F42022(D).evaluate(x), 'name': 'CEC2022 F4', 'optimal': 800, 'bounds': [(-100, 100)] * D},  # All works well
        {'func': lambda x: cec2022.F52022(D).evaluate(x), 'name': 'CEC2022 F5', 'optimal': 900, 'bounds': [(-100, 100)] * D}, # Pincha bien
        {'func': lambda x: cec2022.F62022(D).evaluate(x), 'name': 'CEC2022 F6', 'optimal': 1800, 'bounds': [(-100, 100)] * D}, # Con esta no pincha bien,todos fallan
        {'func': lambda x: cec2022.F72022(D).evaluate(x), 'name': 'CEC2022 F7', 'optimal': 2000, 'bounds': [(-100, 100)] * D},  # Pincha bien
        {'func': lambda x: cec2022.F82022(D).evaluate(x), 'name': 'CEC2022 F8', 'optimal': 2200, 'bounds': [(-100, 100)] * D},  # Pincha bien, WOA y GWO fallan
        {'func': lambda x: cec2022.F92022(D).evaluate(x), 'name': 'CEC2022 F9', 'optimal': 2300, 'bounds': [(-100, 100)] * D},    # Pincha bien
        {'func': lambda x: cec2022.F102022(D).evaluate(x), 'name': 'CEC2022 F10', 'optimal': 2400, 'bounds': [(-100, 100)] * D}, # Baja demasiado, alrededor de 1600
        {'func': lambda x: cec2022.F112022(D).evaluate(x), 'name': 'CEC2022 F11', 'optimal': 2600, 'bounds': [(-100, 100)] * D}, # Pincha bien
        {'func': lambda x: cec2022.F122022(D).evaluate(x), 'name': 'CEC2022 F12', 'optimal': 2700, 'bounds': [(-100, 100)] * D}, # ESO queda 2do
    ]
############ CEC 2021 (10  functions)
elif problem_group == 'CEC2021':
    D = 10
    test_functions = [
        # {'func': lambda x: cec2021.F12021(D).evaluate(x), 'name': 'CEC2021 F1', 'optimal': 100, 'bounds': [(-100, 100)] * D}, ### Not good
        # {'func': lambda x:cec2021.F22021(D).evaluate(x), 'name': 'CEC2021 F2', 'optimal': 1100, 'bounds': [(-100, 100)] * D}, ## Not good
        # {'func': lambda x: cec2021.F32021(D).evaluate(x), 'name': 'CEC2021 F3', 'optimal': 700, 'bounds': [(-100, 100)] * D}, # Pincha bien
        # {'func': lambda x: cec2021.F42021(D).evaluate(x), 'name': 'CEC2021 F4', 'optimal': 1900, 'bounds': [(-100, 100)] * D}, # Pincha bien
        # {'func': lambda x: cec2021.F52021(D).evaluate(x), 'name': 'CEC2021 F5', 'optimal': 1700, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2021.F62021(D).evaluate(x), 'name': 'CEC2021 F6', 'optimal': 1600, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2021.F72021(D).evaluate(x), 'name': 'CEC2021 F7', 'optimal': 2100, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2021.F82021(D).evaluate(x), 'name': 'CEC2021 F8', 'optimal': 2200, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2021.F92021(D).evaluate(x), 'name': 'CEC2021 F9', 'optimal': 2400, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2021.F102021(D).evaluate(x), 'name': 'CEC2021 F10', 'optimal': 2500, 'bounds': [(-100, 100)] * D},
        
    ]
############ CEC 2020 (10  functions)
elif problem_group == 'CEC2020':
    D = 10
    test_functions = [
        # {'func': lambda x: cec2020.F12020(D).evaluate(x), 'name': 'CEC2020 F1', 'optimal': 100, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x:cec2020.F22020(D).evaluate(x), 'name': 'CEC2020 F2', 'optimal': 1100, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F32020(D).evaluate(x), 'name': 'CEC2020 F3', 'optimal': 700, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F42020(D).evaluate(x), 'name': 'CEC2020 F4', 'optimal': 1900, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F52020(D).evaluate(x), 'name': 'CEC2020 F5', 'optimal': 1700, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F62020(D).evaluate(x), 'name': 'CEC2020 F6', 'optimal': 1600, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F72020(D).evaluate(x), 'name': 'CEC2020 F7', 'optimal': 2100, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F82020(D).evaluate(x), 'name': 'CEC2020 F8', 'optimal': 2200, 'bounds': [(-100, 100)] * D},
        # {'func': lambda x: cec2020.F92020(D).evaluate(x), 'name': 'CEC2020 F9', 'optimal': 2400, 'bounds': [(-100, 100)] * D},
        {'func': lambda x: cec2020.F102020(D).evaluate(x), 'name': 'CEC2020 F10', 'optimal': 2500, 'bounds': [(-100, 100)] * D},
        
    ]


# ######### Visualize functions
if visualize == True:
    for test in test_functions:
        plot_func(test['func'], test['bounds'], title=test['name'],dim=len(test['bounds']))
    
    
###### Initialize dictionaries ######
euclidean_distances = {func['name']: {} for func in test_functions}
history = {func['name']: {'scores': {}, 'times': {}, 'objective_values': {}} for func in test_functions}

if entire_set == True:
    ## Whole group
    box_plot_data = {func['name']: {algo: [] for algo in ['ESO', 'WOA', 'GWO', 'HHO', 'PSO', 'GA', 'DE', 'ACO', 'ABC', 'BFO', 'BA', 'BBO', 'CRO', 'EFO', 'FPA', 'CS',  'HS',  'SA', 'TS']} for func in test_functions}
else:    
    ## Single group
    box_plot_data = {func['name']: {algo: [] for algo in ['ESO', 'WOA', 'GWO', 'HHO']} for func in test_functions}
    
###### Run simulations ######
for test in test_functions:
    func = test['func']
    name = test['name']
    bounds = test['bounds']
    
    problem = {
            "obj_func": func,
            "bounds": FloatVar(lb=[b[0] for b in bounds], ub=[b[1] for b in bounds]),
            "minmax": "min",
        }

    # Initialize accumulators
    if entire_set == True:
        accum_scores = {'ESO': [] , 'WOA': [], 'GWO': [], 'HHO':[], 'PSO': [], 'GA': [], 'DE': [], 'ACO': [], 'ABC': [], 'BFO': [], 'BA': [], 'BBO': [], 'CRO': [], 'EFO': [], 'FPA': [], 'CS': [],  'HS': [], 'SA': [], 'TS': []}
    else:    
        accum_scores = {'ESO': [] , 'WOA': [], 'GWO': [], 'HHO':[]}
    accum_times = {key: [] for key in accum_scores}
    
    # Run simulations
    for _ in range(num_simulations):
        # ESO (Electric Storm Optimization)
        start_time = time.time()  # Start measuring time
        eso = ESO(function=func, constraints=[], n_rays=pop_size, iterations=max_iter, bounds=bounds)
        best_score_eso = eso.optimize()[1]
        eso_time = time.time() - start_time  # Finish measuring time
        accum_scores['ESO'].append(best_score_eso)
        accum_times['ESO'].append(eso_time)
        box_plot_data[name]['ESO'].append(best_score_eso)
        history[name]['ESO'] = {
            'scores': accum_scores['ESO'],
            'times': accum_times['ESO'],
            'objective_values': eso.objective_values
        }
                                      
        # WOA (Whale Optimization Algorithm)
        start_time = time.time()
        woa =  WOA.OriginalWOA(epoch=max_iter, pop_size=pop_size,)
        best_score_woa = woa.solve(problem=problem)
        woa_time = time.time() - start_time
        accum_scores['WOA'].append(best_score_woa.target.fitness)
        accum_times['WOA'].append(woa_time)
        box_plot_data[name]['WOA'].append(best_score_woa.target.fitness if isinstance(best_score_woa.target.fitness, np.ndarray) else best_score_woa.target.fitness)
        history[name]['WOA'] = {
            'scores': accum_scores['WOA'],
            'times': accum_times['WOA'],
            'objective_values': woa.history.list_global_best_fit
        }
        
         # GWO (Grey Wolf Optimization)
        start_time = time.time()
        gwo =  GWO.OriginalGWO(epoch=max_iter, pop_size=pop_size)
        best_score_gwo = gwo.solve(problem=problem)
        gwo_time = time.time() - start_time
        accum_scores['GWO'].append(best_score_gwo.target.fitness)
        accum_times['GWO'].append(gwo_time)
        box_plot_data[name]['GWO'].append(best_score_gwo.target.fitness if isinstance(best_score_gwo.target.fitness, np.ndarray) else best_score_gwo.target.fitness)
        history[name]['GWO'] = {
            'scores': accum_scores['GWO'],
            'times': accum_times['GWO'],
            'objective_values': gwo.history.list_global_best_fit
        }
        
        # HHO (Harris Hawks Optimization)
        start_time = time.time()
        hho =   HHO.OriginalHHO(epoch=100, pop_size=50)
        best_score_hho = hho.solve(problem=problem)
        hho_time = time.time() - start_time
        accum_scores['HHO'].append(best_score_hho.target.fitness)
        accum_times['HHO'].append(hho_time)
        box_plot_data[name]['HHO'].append(best_score_hho.target.fitness if isinstance(best_score_hho.target.fitness, np.ndarray) else best_score_hho.target.fitness)
        history[name]['HHO'] = {
            'scores': accum_scores['HHO'],
            'times': accum_times['HHO'],
            'objective_values': hho.history.list_global_best_fit
        }
                
        if entire_set == True:
            # PSO (Particle Swarm Optimization)
            start_time = time.time()
            pso = PSO.OriginalPSO(epoch=max_iter, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4)
            best_score_pso = pso.solve(problem=problem)
            pso_time = time.time() - start_time
            accum_scores['PSO'].append(best_score_pso.target.fitness)
            accum_times['PSO'].append(pso_time)
            box_plot_data[name]['PSO'].append(best_score_pso.target.fitness if isinstance(best_score_pso.target.fitness, np.ndarray) else best_score_pso.target.fitness)
            history[name]['PSO'] = {
                'scores': accum_scores['PSO'],
                'times': accum_times['PSO'],
                'objective_values': pso.history.list_global_best_fit
            }
            
            # GA (Genetic Algorithm)
            start_time = time.time()
            ga = GA.BaseGA(epoch=max_iter, pop_size=pop_size, pc=0.9, pm=0.05)
            best_score_ga = ga.solve(problem=problem)
            ga_time = time.time() - start_time
            accum_scores['GA'].append(best_score_ga.target.fitness)
            accum_times['GA'].append(ga_time)
            box_plot_data[name]['GA'].append(best_score_ga.target.fitness if isinstance(best_score_ga.target.fitness, np.ndarray) else best_score_ga.target.fitness)
            history[name]['GA'] = {
                'scores': accum_scores['GA'],
                'times': accum_times['GA'],
                'objective_values': ga.history.list_global_best_fit
            }
                                        
            # DE (Differential Evolution)
            start_time = time.time()
            de =  DE.OriginalDE(epoch=max_iter, pop_size=pop_size, wf = 0.7, cr = 0.9, strategy = 0)
            best_score_de = de.solve(problem=problem)
            de_time = time.time() - start_time
            accum_scores['DE'].append(best_score_de.target.fitness)
            accum_times['DE'].append(de_time)
            box_plot_data[name]['DE'].append(best_score_de.target.fitness if isinstance(best_score_de.target.fitness, np.ndarray) else best_score_de.target.fitness)
            history[name]['DE'] = {
                'scores': accum_scores['DE'],
                'times': accum_times['DE'],
                'objective_values': de.history.list_global_best_fit
            }
                           
            # ACO (Ant Colony Optimization)
            start_time = time.time()
            aco =  ACOR.OriginalACOR(epoch=max_iter, pop_size=pop_size, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
            best_score_aco = aco.solve(problem=problem)
            aco_time = time.time() - start_time
            accum_scores['ACO'].append(best_score_aco.target.fitness)
            accum_times['ACO'].append(aco_time)
            box_plot_data[name]['ACO'].append(best_score_aco.target.fitness if isinstance(best_score_aco.target.fitness, np.ndarray) else best_score_aco.target.fitness)
            history[name]['ACO'] = {
                'scores': accum_scores['ACO'],
                'times': accum_times['ACO'],
                'objective_values': aco.history.list_global_best_fit
            }
                
            # ABC (Artificial Bee Colony)
            start_time = time.time()
            abc =  ABC.OriginalABC(epoch=max_iter, pop_size=pop_size, n_limits = 50)
            best_score_abc = abc.solve(problem=problem)
            abc_time = time.time() - start_time
            accum_scores['ABC'].append(best_score_abc.target.fitness)
            accum_times['ABC'].append(abc_time)
            box_plot_data[name]['ABC'].append(best_score_abc.target.fitness if isinstance(best_score_abc.target.fitness, np.ndarray) else best_score_abc.target.fitness)
            history[name]['ABC'] = {
                'scores': accum_scores['ABC'],
                'times': accum_times['ABC'],
                'objective_values': abc.history.list_global_best_fit
            }
            
            # BFO (Bacterial Foraging Optimization)
            start_time = time.time()
            bfo = BFO.ABFO(epoch=max_iter, pop_size=pop_size,C_s=0.1, C_e=0.001, Ped=0.01, Ns=4, N_adapt=2, N_split=40)
            best_score_bfo = bfo.solve(problem=problem)
            bfo_time = time.time() - start_time
            accum_scores['BFO'].append(best_score_bfo.target.fitness)
            accum_times['BFO'].append(bfo_time)
            box_plot_data[name]['BFO'].append(best_score_bfo.target.fitness if isinstance(best_score_bfo.target.fitness, np.ndarray) else best_score_bfo.target.fitness)
            history[name]['BFO'] = {
                'scores': accum_scores['BFO'],
                'times': accum_times['BFO'],
                'objective_values': bfo.history.list_global_best_fit
            }
        
            # BA (Bat Algorithm)
            start_time = time.time()
            ba = BA.OriginalBA(epoch=max_iter, pop_size=pop_size,loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0)
            best_score_ba = ba.solve(problem=problem)
            ba_time = time.time() - start_time
            accum_scores['BA'].append(best_score_ba.target.fitness)
            accum_times['BA'].append(ba_time)
            box_plot_data[name]['BA'].append(best_score_ba.target.fitness if isinstance(best_score_ba.target.fitness, np.ndarray) else best_score_ba.target.fitness)
            history[name]['BA'] = {
                'scores': accum_scores['BA'],
                'times': accum_times['BA'],
                'objective_values': ba.history.list_global_best_fit
            }
            
            # BBO (Biogeography-Based Optimization)
            start_time = time.time()
            bbo = BBO.OriginalBBO(epoch=max_iter, pop_size=pop_size, p_m=0.01, n_elites=2)
            best_score_bbo = bbo.solve(problem=problem)
            bbo_time = time.time() - start_time
            accum_scores['BBO'].append(best_score_bbo.target.fitness)
            accum_times['BBO'].append(bbo_time)
            box_plot_data[name]['BBO'].append(best_score_bbo.target.fitness if isinstance(best_score_bbo.target.fitness, np.ndarray) else best_score_bbo.target.fitness)
            history[name]['BBO'] = {
                'scores': accum_scores['BBO'],
                'times': accum_times['BBO'],
                'objective_values': bbo.history.list_global_best_fit
            }
            
            # CRO (Coral Reefs Optimization)
            start_time = time.time()
            cro = CRO.OriginalCRO(epoch=max_iter, pop_size=pop_size, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, gamma_min=0.02, gamma_max=0.2, n_trials=5)
            best_score_cro = cro.solve(problem=problem)
            cro_time = time.time() - start_time
            accum_scores['CRO'].append(best_score_cro.target.fitness)
            accum_times['CRO'].append(cro_time)
            box_plot_data[name]['CRO'].append(best_score_cro.target.fitness if isinstance(best_score_cro.target.fitness, np.ndarray) else best_score_cro.target.fitness)
            history[name]['CRO'] = {
                'scores': accum_scores['CRO'],
                'times': accum_times['CRO'],
                'objective_values': cro.history.list_global_best_fit
            }
            
            # EFO (Electromagnetic Field Optimization)
            start_time = time.time()
            efo = EFO.DevEFO(epoch=max_iter, pop_size=pop_size, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
            best_score_efo = efo.solve(problem=problem)
            efo_time = time.time() - start_time
            accum_scores['EFO'].append(best_score_efo.target.fitness)
            accum_times['EFO'].append(efo_time)
            box_plot_data[name]['EFO'].append(best_score_efo.target.fitness if isinstance(best_score_efo.target.fitness, np.ndarray) else best_score_efo.target.fitness)
            history[name]['EFO'] = {
                'scores': accum_scores['EFO'],
                'times': accum_times['EFO'],
                'objective_values': efo.history.list_global_best_fit
            }
                        
            # FPA (Flower Pollination Algorithm)
            start_time = time.time()
            fpa = FPA.OriginalFPA(epoch=max_iter, pop_size=pop_size, p_s=0.8, levy_multiplier=0.2)
            best_score_fpa = fpa.solve(problem=problem)
            fpa_time = time.time() - start_time
            accum_scores['FPA'].append(best_score_fpa.target.fitness)
            accum_times['FPA'].append(fpa_time)
            box_plot_data[name]['FPA'].append(best_score_fpa.target.fitness if isinstance(best_score_fpa.target.fitness, np.ndarray) else best_score_fpa.target.fitness)
            history[name]['FPA'] = {
                'scores': accum_scores['FPA'],
                'times': accum_times['FPA'],
                'objective_values': fpa.history.list_global_best_fit
            }
            
            # CSA (Cuckoo Search)
            start_time = time.time()
            cs =  CSA.OriginalCSA(epoch=max_iter, pop_size=pop_size, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
            best_score_cs = cs.solve(problem=problem)
            cs_time = time.time() - start_time
            accum_scores['CS'].append(best_score_cs.target.fitness)
            accum_times['CS'].append(cs_time)
            box_plot_data[name]['CS'].append(best_score_cs.target.fitness if isinstance(best_score_cs.target.fitness, np.ndarray) else best_score_cs.target.fitness)
            history[name]['CS'] = {
                'scores': accum_scores['CS'],
                'times': accum_times['CS'],
                'objective_values': cs.history.list_global_best_fit
            }
            
            # HS (Harmony Search)
            start_time = time.time()
            hs = HS.OriginalHS(epoch=max_iter, pop_size=pop_size)
            best_score_hs = hs.solve(problem=problem)
            hs_time = time.time() - start_time
            accum_scores['HS'].append(best_score_hs.target.fitness)
            accum_times['HS'].append(hs_time)
            box_plot_data[name]['HS'].append(best_score_hs.target.fitness if isinstance(best_score_hs.target.fitness, np.ndarray) else best_score_hs.target.fitness)
            history[name]['HS'] = {
                'scores': accum_scores['HS'],
                'times': accum_times['HS'],
                'objective_values': hs.history.list_global_best_fit
            }
                            
            # SA (Simulated Annealing)
            start_time = time.time()
            sa =  SA.OriginalSA(epoch=max_iter, pop_size=pop_size, temp_init = 100, step_size = 0.1)
            best_score_sa = sa.solve(problem=problem)
            sa_time = time.time() - start_time
            accum_scores['SA'].append(best_score_sa.target.fitness)
            accum_times['SA'].append(sa_time)
            box_plot_data[name]['SA'].append(best_score_sa.target.fitness if isinstance(best_score_sa.target.fitness, np.ndarray) else best_score_sa.target.fitness)
            history[name]['SA'] = {
                'scores': accum_scores['SA'],
                'times': accum_times['SA'],
                'objective_values': sa.history.list_global_best_fit
            }
                                    
            # TS (Tabu Search)
            start_time = time.time()
            ts = TS.OriginalTS(epoch=max_iter, pop_size=2,tabu_size = 5, neighbour_size = 10, perturbation_scale = 0.05)
            best_score_ts = ts.solve(problem=problem)
            ts_time = time.time() - start_time
            accum_scores['TS'].append(best_score_ts.target.fitness)
            accum_times['TS'].append(ts_time)
            box_plot_data[name]['TS'].append(best_score_ts.target.fitness if isinstance(best_score_ts.target.fitness, np.ndarray) else best_score_ts.target.fitness)
            history[name]['TS'] = {
                'scores': accum_scores['TS'],
                'times': accum_times['TS'],
                'objective_values': ts.history.list_global_best_fit
                
            }
                
        ##################### Discarded
        # # FA (Fireworks Algorithm)
        # start_time = time.time()
        # fa = FA.OriginalFA(epoch=max_iter, pop_size=pop_size, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)
        # best_score_fa = fa.solve(problem=problem)
        # fa_time = time.time() - start_time
        # accum_scores['FA'].append(best_score_fa.target.fitness)
        # accum_times['FA'].append(fa_time)
        # box_plot_data[name]['FA'].append(best_score_fa.target.fitness if isinstance(best_score_fa.target.fitness, np.ndarray) else best_score_fa.target.fitness)
        # history[name]['FA'] = {
        #     'scores': accum_scores['FA'],
        #     'times': accum_times['FA'],
        #     'objective_values': fa.history.list_global_best_fit
        # }
                
        # # GOA (Grasshopper Optimization Algorithm)
        # start_time = time.time()
        # goa = GOA.OriginalGOA(epoch=max_iter, pop_size=pop_size,c_min=0.00004, c_max=1.0)
        # best_score_goa = goa.solve(problem=problem)
        # goa_time = time.time() - start_time
        # accum_scores['GOA'].append(best_score_goa.target.fitness)
        # accum_times['GOA'].append(goa_time)
        # box_plot_data[name]['GOA'].append(best_score_goa.target.fitness if isinstance(best_score_goa.target.fitness, np.ndarray) else best_score_goa.target.fitness)
        # history[name]['GOA'] = {
        #     'scores': accum_scores['GOA'],
        #     'times': accum_times['GOA'],
        #     'objective_values': goa.history.list_global_best_fit
        # }
                               
        # # NRO (Nuclear Reaction Optimization) Too slow
        # start_time = time.time()
        # nro = NRO.OriginalNRO(epoch=max_iter, pop_size=pop_size, )
        # best_score_nro = nro.solve(problem=problem)
        # nro_time = time.time() - start_time
        # accum_scores['NRO'].append(best_score_nro.target.fitness)
        # accum_times['NRO'].append(nro_time)
        # box_plot_data[name]['NRO'].append(best_score_nro.target.fitness if isinstance(best_score_nro.target.fitness, np.ndarray) else best_score_nro.target.fitness)
        # history[name]['NRO'] = {
        #     'scores': accum_scores['NRO'],
        #     'times': accum_times['NRO'],
        #     'objective_values': nro.history.list_global_best_fit
        # }
              
             
    # Calculating average scores and times
    for algo in accum_scores:
        history[name]['scores'][algo] = np.mean(accum_scores[algo])
        history[name]['times'][algo] = np.mean(accum_times[algo])
        
###### Print results ########
data_list = []
history_records = []

for func_name, algorithms in history.items():
    for algo_name, algo_data in algorithms.items():
        # Verificar si 'scores', 'times' y 'objective_values' existen y tienen la misma longitud
        if 'scores' in algo_data and 'times' in algo_data and 'objective_values' in algo_data:
            num_entries = min(len(algo_data['scores']), len(algo_data['times']), len(algo_data['objective_values']))
            for i in range(num_entries):
                history_records.append({
                    'Function': func_name,
                    'Algorithm': algo_name,
                    'Score': algo_data['scores'][i],
                    'Time': algo_data['times'][i],
                    'Objective Value': algo_data['objective_values'][i]
                })

# Convertir la lista de registros a un DataFrame
history_raw = pd.DataFrame(history_records)
# Formatear la fecha y hora actual para el nombre del archivo
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Results', exist_ok=True)
filename = f"Results/{problem_group}_{current_datetime}_history_raw.xlsx"
# Exportar el DataFrame a un archivo Excel
history_raw.to_excel(filename, index=False)

print(f"Archivo guardado como: {filename}")

success_threshold = 0.001  # Umbral para definir un éxito 
for name in history:
    known_optimum = next(item['optimal'] for item in test_functions if item['name'] == name)
    print(f"{name} Function")

    for algo in history[name]['scores']:
        scores = history[name][algo]['scores']
        best_score = min(scores, key=lambda x: abs(x - known_optimum))
        worst_score = max(scores, key=lambda x: abs(x - known_optimum))
        std_dev = np.std(scores, ddof=1)
        avg_score_numeric = history[name]['scores'][algo]
        distance = np.linalg.norm(known_optimum - avg_score_numeric)  
        euclidean_distances[name][algo] = distance
        num_successes = sum(abs(score - known_optimum) <= success_threshold for score in scores)
        success_ratio = num_successes / len(scores)
        avg_score_str = '{:.10e}'.format(avg_score_numeric)
        std_dev = '{:.10e}'.format(std_dev) 
        avg_time = '{:.5e}'.format(history[name]['times'][algo])
        distance_str = '{:.10e}'.format(distance)
        # Append results to data_list
        data_list.append({
            'Function': name,
            'Algorithm': algo,
            'Best Score': best_score,
            'Worst Score': worst_score,
            'Average Score': avg_score_numeric,
            'Standard Deviation': std_dev,
            'Euclidean Distance': distance_str,
            'Success Ratio': success_ratio,
            'Average Time': avg_time,
            
        })
        print(f"  {algo}: Average Best Score = {avg_score_str}, Best Score = {avg_score_str}, Worst Score = {worst_score},  Std Dev = {std_dev}, Success Ratio = {success_ratio:.5f}, Euclidean Distance = {distance_str}, Average Time = {avg_time}")
   
    print("\n")

# Crear DataFrame a partir de la lista de diccionarios
history_df = pd.DataFrame(data_list)

# Formatear la fecha y hora actual para el nombre del archivo
filename = f"Results/{problem_group}_{current_datetime}_history_processed.xlsx"

# Exportar el DataFrame a un archivo CSV
history_df.to_excel(filename, index=False)
    
#################### Plot results ####################
data = {algo: [euclidean_distances[func][algo] for func in euclidean_distances] for algo in list(euclidean_distances.values())[0]}
df = pd.DataFrame(data, index=euclidean_distances.keys())

# Order the algorithms by their average performance
df_ranked = df.rank(axis=1, method='min', ascending=True)

########## Rankin heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_ranked, annot=True, cmap="Oranges", cbar_kws={'label': 'Ranking (Lower is Better)'}, annot_kws={"size": 14})
plt.title('Algorithm Rankings', fontsize=18)
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Function', fontsize=14)
plt.yticks(rotation=0, fontsize=14, va="center", ticks=np.arange(0.5, len(df_ranked.index), 1), labels=df_ranked.index)
plt.xticks(rotation=45, fontsize=14, ha="right", ticks=np.arange(0.5, len(df_ranked.columns), 1), labels=df_ranked.columns)
plt.tight_layout()  
plt.show()

########## Results boxplot    
num_functions = len(box_plot_data)
nrows = 2
ncols = (num_functions + 1) // 2 if num_functions % 2 != 0 else num_functions // 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))

if nrows > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, (func_name, data) in enumerate(box_plot_data.items()):
    ax = axes[i]
    ax.boxplot([data[algo] for algo in data],
               labels=[algo for algo in data],
               notch=True)
    ax.set_title(f'Function {func_name}')
    ax.set_ylabel('Score' if i % ncols == 0 else "", fontsize=18)
    ax.set_xlabel('Algorithm', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticklabels([algo for algo in data], rotation=45)  # Rotar etiquetas

if num_functions % 2 != 0:
    axes[-1].axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Ajustar el espacio entre subplots
plt.tight_layout()
plt.show()
           
############## Plot optimization progress
#### Line styles
styles = {
    'ESO': {'linestyle': '-'},
    'WOA': {'linestyle': '--'},
    'GWO': {'linestyle': '-.'},
    'HHO': {'linestyle': '-.'},
    'PSO': {'linestyle': ':'},
    'GA': {'linestyle': '--'},
    'DE': {'linestyle': '-'},
    'ACO': {'linestyle': '--'},
    'ABC': {'linestyle': '--'},
    'BFO': {'linestyle': '-.'},
    'BA': {'linestyle': ':'},
    'BBO': {'linestyle': ':'},
    'CRO': {'linestyle': '-'},
    'EFO': {'linestyle': '-.'},
    'FPA': {'linestyle': '--'},
    'CS': {'linestyle': '-'},
    'HS': {'linestyle': '-.'},
    'SA': {'linestyle': '-'},
    'TS': {'linestyle': '--'},
    
}

num_functions = len(history)
num_cols = (num_functions + 1) // 2  # Número de columnas necesario para acomodar todas las funciones en dos filas

# Crear figuras con dos filas y un número variable de columnas
fig1, axs1 = plt.subplots(2, num_cols, figsize=(12 * num_cols, 12))
# fig2, axs2 = plt.subplots(2, num_cols, figsize=(12 * num_cols, 12))
fig3, axs3 = plt.subplots(2, num_cols, figsize=(12 * num_cols, 12))

# Aplanar los ejes para un manejo más sencillo
axs1 = axs1.flatten()
# axs2 = axs2.flatten()
axs3 = axs3.flatten()

for idx, (name, algo_data) in enumerate(history.items()):
    # Configurar los ejes para cada función
    ax1 = axs1[idx]
    # ax2 = axs2[idx]
    ax3 = axs3[idx]
    
    ax1.set_title(name, fontsize=16)
    # ax2.set_title(name, fontsize=16)
    ax3.set_title(name, fontsize=16)
    

    # Progreso de Optimización
    for algo in algo_data:
        if 'objective_values' in algo_data[algo]:
            style = styles.get(algo, {'linestyle': '-', 'marker': None})
            ax1.plot(algo_data[algo]['objective_values'], label=algo, **style)
            
    # Progreso Normalizado Hacia el Óptimo
    optimal_value = next(item['optimal'] for item in test_functions if item['name'] == name)
    # for algo in algo_data:
    #     if 'objective_values' in algo_data[algo]:
    #         style = styles.get(algo, {'linestyle': '-', 'marker': None})
    #         values = algo_data[algo]['objective_values']
    #         normalized_progress = [(v - optimal_value) / (max(values) - optimal_value) for v in values]
    #         ax2.plot(normalized_progress, label=algo, **style)
            

    # Ratio de Convergencia
    for algo in algo_data:
        if 'objective_values' in algo_data[algo]:
            style = styles.get(algo, {'linestyle': '-', 'marker': None})
            values = algo_data[algo]['objective_values']
            convergence_ratio = [(optimal_value - value) / (optimal_value - values[0]) if values[0] != optimal_value else 1 for value in values]
            ax3.plot(convergence_ratio, label=algo, **style)
            
# Ocultar ejes vacíos si hay un número impar de funciones
for ax in [axs1, axs3]:
    for i in range(num_functions, len(ax)):
        ax[i].axis('off')

# Configurar ejes y leyendas
for ax in axs1[-num_cols:]:
    ax.set_xlabel("Iteration", fontsize=18)
# for ax in axs2[-num_cols:]:
#     ax.set_xlabel("Iteration", fontsize=18)
for ax in axs3[-num_cols:]:
    ax.set_xlabel("Iteration", fontsize=18)

fig1.suptitle("Optimization Progress", fontsize=22)
# fig2.suptitle("Normalized Optimization Progress", fontsize=22)
fig3.suptitle("Convergence Ratio", fontsize=22)

handles, labels = axs1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncol=len(algo_data), fontsize=12)
# fig2.legend(handles, labels, loc='lower center', ncol=len(algo_data), fontsize=12)
fig3.legend(handles, labels, loc='lower center', ncol=len(algo_data), fontsize=12)

fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

############# Results heatmap    
scores_df = pd.DataFrame({func: {algo: history[func]['scores'][algo] for algo in history[func]['scores']} for func in history}).T
times_df = pd.DataFrame({func: {algo: history[func]['times'][algo] for algo in history[func]['times']} for func in history}).T

# Crear el heatmap de los datos
plt.figure(figsize=(12, 8))
sns.heatmap(scores_df, annot=True, cmap="Oranges", fmt=".3e", cbar=True)
plt.title('Results Heatmap', fontsize=18)
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Function', fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(rotation=0, fontsize=14)  # Rotar las etiquetas del eje y
plt.show()

############ Algortihms execution time heatmap

algorithms = list(accum_scores.keys())
times_df = pd.DataFrame({func['name']: {algo: history[func['name']]['times'][algo] for algo in algorithms} for func in test_functions})
times_df = times_df.T

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(times_df, annot=True, cmap="Oranges", fmt=".3f", cbar_kws={'label': 'Average Time (Seconds)'}, annot_kws={"size": 12})
plt.title('Average Computation Times of Optimization Algorithms', fontsize=16)
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Function', fontsize=14)
plt.yticks(rotation=0, fontsize=14, va="center", ticks=np.arange(0.5, len(df_ranked.index), 1), labels=df_ranked.index)
plt.xticks(rotation=45, fontsize=14, ha="right", ticks=np.arange(0.5, len(df_ranked.columns), 1), labels=df_ranked.columns)
plt.tight_layout()  
plt.show()

# Crear gráfico de líneas
plt.figure(figsize=(12, 8))
for algorithm in algorithms:
    style = styles.get(algo, {'linestyle': '-', 'marker': None})
    plt.plot(times_df[algorithm], label=algorithm, **style)

plt.title('Execution Time of Algorithms Across Functions', fontsize=16)
plt.xlabel('Function', fontsize=14)
plt.ylabel('Average Time (Seconds)', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha="right")
plt.legend(title='Algorithms')
plt.tight_layout()
plt.show()

##### Rankin heatmap based on time
algorithms = list(accum_scores.keys())
# Crear DataFrame de tiempos de ejecución
times_df = pd.DataFrame({func['name']: {algo: history[func['name']]['times'][algo] for algo in algorithms} for func in test_functions})

# Transponer DataFrame para que las funciones sean filas y los algoritmos columnas
times_df = times_df.T

# Calcular el ranking para cada función basado en el tiempo de ejecución
# Los rangos más bajos (1) son los mejores (menor tiempo de ejecución)
ranking_df = times_df.rank(axis=1, method='min', ascending=True)

# Crear heatmap de ranking
plt.figure(figsize=(12, 8))
sns.heatmap(ranking_df, annot=True, cmap="Oranges", fmt=".0f", cbar_kws={'label': 'Ranking (Lower is Better)'}, annot_kws={"size": 14})
plt.title('Ranking of Optimization Algorithms by Computation Time', fontsize=16)
plt.xlabel('Algorithm', fontsize=16)
plt.ylabel('Function', fontsize=16)
plt.yticks(rotation=0, fontsize=14, va="center", ticks=np.arange(0.5, len(df_ranked.index), 1), labels=df_ranked.index)
plt.xticks(rotation=45, fontsize=14, ha="right", ticks=np.arange(0.5, len(df_ranked.columns), 1), labels=df_ranked.columns)
plt.tight_layout()  
plt.show()

####### Global ranking considering both score and time
df_time_ranked = times_df.rank(axis=1, method='min', ascending=True)

# Suma los rankings de precisión y tiempo para obtener un score combinado
df_combined_scores = df_ranked + df_time_ranked

# Calcula el ranking global basado en el score combinado
df_global_ranked = df_combined_scores.rank(axis=1, method='min', ascending=True)

# Crear heatmap del ranking global
plt.figure(figsize=(12, 8))
sns.heatmap(df_global_ranked, annot=True, cmap="Oranges", cbar_kws={'label': 'Global Ranking (Lower is Better)'}, annot_kws={"size": 14})
plt.title('Global Algorithm Rankings (Precision-Time)', fontsize=18)
plt.xlabel('Algorithm', fontsize=18)
plt.ylabel('Function', fontsize=18)
plt.yticks(rotation=0, fontsize=14, va="center", ticks=np.arange(0.5, len(df_ranked.index), 1), labels=df_ranked.index)
plt.xticks(rotation=45, fontsize=14, ha="right", ticks=np.arange(0.5, len(df_ranked.columns), 1), labels=df_ranked.columns)
plt.tight_layout()  
plt.show()

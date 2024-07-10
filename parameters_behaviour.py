import numpy as np
import matplotlib.pyplot as plt
from ESO1 import ESO
import BMF

# Lista de funciones y sus respectivos límites
functions = [
    (BMF.Leon.function, [BMF.Leon.bounds for _ in range(2)]),      
    (BMF.Stepint.function, [BMF.Stepint.bounds for _ in range(2)]),
    (BMF.WayburnSeader2.function, [BMF.RosenbrockModified.bounds for _ in range(2)]),     
    (BMF.Damavandi.function, [BMF.Damavandi.bounds for _ in range(2)]),
    (BMF.XinSheYang1.function, [BMF.XinSheYang1.bounds for _ in range(2)]),
    (BMF.Zimmerman.function, [BMF.Zimmerman.bounds for _ in range(2)])
]

# Crear la figura con 6 subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Nombres de las funciones para los títulos de los gráficos
function_names = ['F6', 'F20', 'F24', 'F33', 'F46', 'F50']

# Placeholder for lines to add to the legend
lines_labels = []

for i, (function, bounds) in enumerate(functions):
    # Instanciar y optimizar con ESO
    eso = ESO(function, bounds=bounds)
    best_position, best_value = eso.optimize()

    # Normalización de los datos recopilados
    intensity_normalized = np.array(eso.field_intensity_history) / max(eso.field_intensity_history)
    resistance_normalized = np.array(eso.field_resistance_history) / max(eso.field_resistance_history)
    elasticity_normalized = np.array(eso.field_elasticity_history) / max(eso.field_elasticity_history)
    global_best_normalized = np.array(eso.objective_values) / max(eso.objective_values)
    storm_power = np.array(eso.storm_power_history) / max(eso.storm_power_history)

    # Seleccionar el subplot correspondiente
    ax = axs[i // 3, i % 3]

    # Configuración de la gráfica
    line0, = ax.plot(storm_power,label='Storm Power', linewidth=2)
    line1, = ax.plot(intensity_normalized, label='Field Intensity', linewidth=2)
    line2, = ax.plot(resistance_normalized, label='Field Resistance', linewidth=2)
    line3, = ax.plot(elasticity_normalized, label='Field Elasticity', linewidth=2)
    line4, = ax.plot(global_best_normalized, label='Global Best', linewidth=3, linestyle='--')
    
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Normalized Value', fontsize=16)
    ax.set_title(function_names[i], fontsize=18)
    ax.grid(True)

    # Save lines for the legend
    if i == 0:
        lines_labels = [(line0, 'Storm Power'), (line1, 'Field Intensity'), (line2, 'Field Resistance'),
                        (line3, 'Field Elasticity'), (line4, 'Global Best')]

# Create a single legend for the entire figure
fig.legend(*zip(*lines_labels), loc='upper center', ncol=5, fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

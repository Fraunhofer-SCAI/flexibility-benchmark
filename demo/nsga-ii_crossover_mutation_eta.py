"""
Demonstrates how to estimate the perturbation in parameters given a certain ETA parameter value for the mate and mutate
operators used in experiments/baseline_adaption_nsga_ii.py.
"""

import numpy as np
from deap import base, creator, tools

import random

# Process parameters.
n_process_params = 3
range_process_params = [
    [0.1, 5.0],  # Range for cutting speed.
    [-0.5, 1.0],  # Range for cutting angle.
    [1.0e-6, 1.0e-3],  # Range for cutting depth.
]
lower_bounds = [x[0] for x in range_process_params]
upper_bounds = [x[1] for x in range_process_params]

# Eta values
eta_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 190, 200]

# Initialize deap toolbox.
toolbox = base.Toolbox()

# Create FitnessMin class for multiple (four) objectives.
creator.create('FitnessMin', base.Fitness, weights=(-1.0,) * 4)

# Create Individual class, that is a list with fitness class FitnessMin.
creator.create('Individual', list, fitness=creator.FitnessMin)

# Function to initialize an individual.
init_ind = lambda: [
    random.uniform(range_process_params[idx][0], range_process_params[idx][1]) for idx in range(n_process_params)
]

# Individual (list of float values).
toolbox.register('individual', tools.initIterate, creator.Individual, init_ind)

# Demonstration for mutation.
for eta in eta_values:
    toolbox.register(
        'mutate', tools.mutPolynomialBounded, low=lower_bounds, up=upper_bounds, eta=eta, indpb=1.0 / n_process_params
    )

    ind = toolbox.individual()
    offspring = [toolbox.mutate(toolbox.clone(ind))[0] for _ in range(10000)]
    avg_perturbation = np.mean([np.abs(np.array(ind) - np.array(off)) for off in offspring], axis=0)

    print('Average perturbation for mutation when eta=%d: %s' % (eta, avg_perturbation))

# Demonstration for crossover.
for eta in eta_values:
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=lower_bounds, up=upper_bounds, eta=eta)

    parent1 = toolbox.individual()
    parent2 = toolbox.individual()

    off1, off2 = toolbox.mate(toolbox.clone(parent1), toolbox.clone(parent2))

    print('\nCrossover with eta=%d:' % eta)
    print('parent1: %s' % parent1)
    print('parent2: %s' % parent2)
    print('off1: %s' % off1)
    print('off2: %s' % off2)

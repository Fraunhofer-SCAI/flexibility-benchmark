"""
Demonstrate how to calculate the range for the objective values for the Oxley model, given known ranges
for the process parameters and for the output forces. For different process parameters ranges and tolerance
values for the output forces, the same logic can be applied.
"""

import numpy as np

# Relevant process parameters ranges.
cutting_speed_min = 0.1
cutting_speed_max = 5.0
cutting_depth_min = 1.0e-6
cutting_depth_max = 1.0e-3

# Output forces tolerances (absolute values).
Fc_min = 0
Fc_max = 500
Ft_min = 0
Ft_max = 500


# Functions for production time, tool wear, and number of layers.
def prod_time(speed, n_layers):
    return (1.0 / speed) * n_layers


def tool_wear(speed, n_layers, Fc, Ft):
    return (speed * np.exp(abs(Fc)) + 0.1 * speed * np.exp(abs(Ft))) * n_layers


def n_layers(cutting_depth):
    return int(np.ceil(1.0 / cutting_depth))


# Minimum n_layers is achieved with maximum cutting_depth.
n_layers_min = n_layers(cutting_depth_max)

# Maximum n_layers is achieved with minimum cutting_depth.
n_layers_max = n_layers(cutting_depth_min)

print('n_layers_min=%f' % n_layers_min)
print('n_layers_max=%f' % n_layers_max)

# Minimum production time is achieved with maximum cutting_speed and minimum n_layers.
prod_time_min = prod_time(cutting_speed_max, n_layers_min)

# Maximum production time is achieved with minimum cutting_speed and maximum n_layers.
prod_time_max = prod_time(cutting_speed_min, n_layers_max)

print('prod_time_min=%f' % prod_time_min)
print('prod_time_max=%f' % prod_time_max)

# Minimum tool_wear is achieved with minimum cutting_speed, n_layers, Fc, and Ft.
tool_wear_min = tool_wear(cutting_speed_min, n_layers_min, Fc_min, Ft_min)

# Maximum tool_wear is achieved with maximum cutting_speed, n_layers, Fc, and Ft.
tool_wear_max = tool_wear(cutting_speed_max, n_layers_max, Fc_max, Ft_max)

print('tool_wear_min=%f' % tool_wear_min)
print('tool_wear_max=%E' % tool_wear_max)

from flexbench.benchmarks.oxley_model_iwm import ExtendedOxleyTask
from random import uniform
import math

# 1 - Parameter ranges had to be adjusted so that solutions generated are feasible and don't cause the simulation to fail.
# 2 - We are not optimizing the cutting_width, but it did influence the output forces. I set it to 1.6e-4 inside the step function so that solutions
#     are feasible.
# 3 - I set the total amount of material to be removed to 1mm. Depending on the cutting_depth, a number of layers is required. The final layer may be
#     thinner than the value from total_depth, but the output from it is not being considered.
# 4 - The material parameters for aluminium make the simulation fail, so I left it out for now.
# 5 - Some random solutions still cause the simulation to fail sometimes (ValueError, TypeError).
# 6 - The two last issues above also happen with the original code I tested (OxleyPython), so I assume it's how the simulation works. However,
#     when I compare the output from the current simulation with the original, there are some slight differences. I assume it's some precision issue
#     depending on the libraries used, but you could check it if you think it's relevant at some point.

total_depth = 1.0  # Total depth of material to be removed (mm)
total_length = 1.0  # Total length of material to be removed (m)

# material_params = [273.15,298.15,8000.0,0.9,0.9,553.1e+6,600.8e+6,0.234,0.0134,1.0,1733.17,1.0] # From original code.
# material_params = [273.15,300.0,2700.0,0.9,0.9,1.67e+8,5.96e+8,0.551,0.001,0.859,893.0,1.0] # Aluminium
material_params = [273.15, 300.0, 7860.0, 0.9, 0.9, 7.92e8, 5.10e8, 0.26, 0.014, 1.03, 1790.0, 1.0]  # Steel
# material_params = [273.15,300.0,17600.0,0.9,0.9,1.51e+9,1.77e+8,0.12,0.016,1.0,1723.0,1.0] # Tungsten Alloy
# material_params = [273.15,300.0,7860.0,0.9,0.9,5.82e+8,4.65e+8,0.325,0.008,1.3,1790.0,1.0] # Steel Dummy
# material_params = [273.15,300.0,8242.0,0.9,0.9,9.28e+8,9.79e+8,0.245847,0.0056,1.80073,1623.15,0.001] #Inconel 718

task = ExtendedOxleyTask(material_params, total_depth, total_length)

cutting_speed = uniform(0.1, 5.0)  # m/s
cutting_angle = uniform(-0.5, 1.0)  # radians
cutting_depth = uniform(1e-6, 1e-3)  # mm

# Solution that made the simulation go into an infinite loop.
# Limiting the iterations in _compute_tc solves it and raises and error we already deal with (ValueError).
# cutting_speed = 3.7925705817149957
# cutting_angle = -0.2910921746453609
# cutting_depth = 1.0690237984867492e-06

print(
    'Solution:\ncutting_speed (m/s) = %f\ncutting_angle (radians) = %f\ncutting_depth (mm) = %f\n'
    % (cutting_speed, cutting_angle, cutting_depth)
)

shear_angle, Fc, Ft, tc, n_layers = task.extOxley.step([cutting_speed, cutting_angle, cutting_depth])
print(
    'Output:\nshear_angle (degrees) = %f\nFc (N) = %f\nFt (N) = %f\ntc (mm)=%f\nn_layers = %d\n'
    % (shear_angle, Fc, Ft, tc, n_layers)
)

feasibility = task.success_criterion(cutting_speed, Fc, Ft)
performance = task.performance_measure(cutting_speed, Fc, Ft, n_layers, total_length)
print('Feasibility:\nSpeedFeas=%s\nForceFeas=%s\n' % (feasibility[0], feasibility[1]))
print(
    'Performance measure:\nproduction_time (s) = %f\ntool_wear = %f\nFc (N) = %f\nFt (N) = %f'
    % (performance[0], performance[1], performance[2], performance[3])
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
These are the source files of the implementation of the Oxley's machining model
using the LMFIT library in Python.

This work is related to the PhD thesis of Maxime Dawoua Kaoutoing:
Maxime. D. Kaoutoing, Contributions à la modélisation et la simulation de la coupe des métaux:
    vers un outil d'aide à la surveillance par apprentissage, PhD Thesis, Toulouse University, 2020

@author: Olivier Pantalé
"""
import math, csv, itertools, collections
import numpy as np
import lmfit

from abc import ABC, abstractmethod


class ExtendedOxleySimulator(object):
    """
    ### Description

    The Extended oxley simulator simulates the stepwise removal of material from a workpiece
    by a cutting tool based on the extended Oxley's machining model.

    ### Process parameter space

    There are four process parameters that can be varied per step:

    | Parameter | Description     | Unit    | Suggested Range                             | Scale of Perturbation |
    |-----------|-----------------|---------|---------------------------------------------|-----------------------|
    | speed     | Cutting speed   | m/sec   | 0.1 to 5.0 m/s                              | Around 0.01           |
    | angle     | Tool rake angle | radians | -0.5 to 1.0 radians                         | Around 0.01           |
    | width     | Width of cut    | mm      | Fixed = 1.6e-4 mm                           | -                     |
    | depth     | Depth of cut    | mm      | 1.0e-6 to 1.0e-3 mm                         | Around 1.0e-5         |

    ### Observation space

    After each step, the simulator returns the following values:

    | Observation    | Description                                | Unit    | Min  | Max                |
    |----------------|--------------------------------------------|---------|------|--------------------|
    | shear_angle    | The angle at which the chip will separate                                        |
    |                  from the work material during cutting      | radians | -    | -                  |
    | Fc             | Chip formation force in cutting direction  | N       | 0    | 500  [dummy value] |
    | Ft             | Chip formation force in thrust direction   | N       | 0    | 500  [dummy value] |
    | tc             | Chip thickness                             | mm      | 0    | -                  |
    | n_layers       | Layers needed to remove all material       | -       | -    | -                  |

    ### Material parameter space

    The following material parameters of the workpiece can be set
    when creating an instance of the simulator:

    | Parameter | Description                                      | Unit   |
    |-----------|--------------------------------------------------|--------|
    | T0        | Initial temperature                              | Kelvin |
    | Tw        | Ambient temperature                              | Kelvin |
    | Tm        | Melting temperature                              | Kelvin |
    | rho       | Density                                          | km/m^3 |
    | eta       | Temperature averaging factor used in computation
                  of temperature along shear plane                 | -      |
    | psi       | Temperature averaging factor used in computation
                  of tool-chip interface temperature               | -      |
    | jc_A      | A coefficient in the Johnson-Cook law            | Pa     |
    | jc_B      | B coefficient in the Johnson-Cook law            | Pa     |
    | jc_n      | n coefficient in the Johnson-Cook law            | -      |
    | jc_C      | C coefficient in the Johnson-Cook law            | -      |
    | jc_m      | m coefficient in the Johnson-Cook law            | -      |
    | jc_Epsp0  | epislon dot coefficient for the Johnson-Cook law | 1/s    |

    """

    def __init__(self, material_parameters, total_depth, total_length):
        # Set material parameters of the working piece.
        # Here, total_depth and total_length are fixed and won't change the returned forces.
        self.material_parameters = material_parameters
        self.total_depth = total_depth
        self.total_length = total_length

        self.T0 = self.material_parameters[0]  # Initial temperature (Kelvin)
        self.Tw = self.material_parameters[1]  # Ambient temperature (Kelvin)
        self.rho = self.material_parameters[2]  # Density (km/self.m^3)
        self.eta = self.material_parameters[3]  # Value of the self.eta parameter
        self.psi = self.material_parameters[4]  # Value of the self.psi parameter
        self.jc_A = self.material_parameters[5]  # self.A coefficient for the Johnson-Cook law (Pa)
        self.jc_B = self.material_parameters[6]  # self.B coefficient for the Johnson-Cook law (Pa)
        self.jc_n = self.material_parameters[7]  # self.n coefficient for the Johnson-Cook law
        self.jc_C = self.material_parameters[8]  # self.C coefficient for the Johnson-Cook law
        self.jc_m = self.material_parameters[9]  # self.m coefficient for the Johnson-Cook law
        self.Tm = self.material_parameters[10]  # Melting temperature (Kelvin)
        self.jc_Epsp0 = self.material_parameters[11]  # epislon dot coefficient for the Johnson-Cook law

        # Set state of simulator to inital state.
        self.reset()

    def reset(self):
        """Reset workpiece to initial state"""
        self._n_calls = 0  # Number of steps that were called.
        self._removed_material = 0

    def step(self, process_parameters):
        """Simulate the removal of a chip."""

        # Reset state.
        self.reset()

        # Set process parameters.
        self.cutting_speed = process_parameters[0]
        self.cutting_angle = process_parameters[1]
        self.cutting_width = 1.6e-4
        self.cutting_depth = process_parameters[2]

        # Steps needed to remove the total amount of material needed given cutting depth.
        # Actually, only one step is done here as the return values do not change with each step in this simulation.
        self.n_layers = int(np.ceil(self.total_depth / self.cutting_depth))

        # Computes the mchip value from cutting parameters
        self.mchip = self.rho * self.cutting_speed * self.cutting_depth * self.cutting_width

        delta = self._optimize()
        Toint, kchip, sigmaN, sigmaNmax, Fc, Ft, t2 = self._compute_Toint_Kchip(self.COF, self.phiF, delta)

        # C0 = round(C0, 3)
        # delta = round(delta, 3)

        shear_angle = round(self.phiF * 180 / math.pi, 1)
        Fc = round(Fc, 1)
        Ft = round(Ft, 1)
        tc = round(t2 * 1000, 2)

        # Update state.
        # Does as if cutting were done in n_layers.
        self._removed_material = self._removed_material + self.cutting_depth * self.n_layers
        self._n_calls = self._n_calls + self.n_layers

        return shear_angle, Fc, Ft, tc, self.n_layers

    def _johnson_cook(self, eps, epsp, T):
        """Computes the Johnson-Cook equivalent stress"""
        return (
            (self.jc_A + self.jc_B * eps**self.jc_n)
            * (1.0 + self.jc_C * math.log(epsp / self.jc_Epsp0))
            * (1.0 - ((T - self.Tw) / (self.Tm - self.Tw)) ** self.jc_m)
        )

    def _k_law(self, T):
        """Computes the K law"""
        return 52.61 - 0.0281 * (T - self.T0)

    def _cp_law(self, T):
        """Computes the CP law"""
        return 420 + 0.504 * (T - self.T0)

    def _compute_AB(self, lAB, phi, Vs, EpsAB, EpspAB):
        """Computes the TAB temperature, returns 0 if TAB  >  self.Tm."""
        TAB_precision = 1e-3  # Precision on the evaluation of TAB
        # Sets TAB equal to self.Tw
        TAB = self.Tw
        # Initialisation of the max number of loops
        maxLoops = 1000
        # Evaluates the temperature due to plastic deformation
        while maxLoops > 0:
            # Computes Cp and K for TAB
            Cp = self._cp_law(TAB)
            K = self._k_law(TAB)
            # Computes the flow stress of the material
            kAB = (1.0 / math.sqrt(3)) * self._johnson_cook(EpsAB, EpspAB, TAB)
            # Computes the Fs value
            Fs = kAB * lAB * self.cutting_width
            # Computes coefficient RT
            RTtanPhi = math.tan(phi) * (self.rho * Cp * self.cutting_speed * self.cutting_depth) / K
            # Computes beta
            if RTtanPhi > 10:
                beta = 0.3 - 0.15 * math.log10(RTtanPhi)
            else:
                beta = 0.5 - 0.35 * math.log10(RTtanPhi)
            # Computes the delta T
            deltaTsz = ((1.0 - beta) * Fs * Vs) / (self.mchip * Cp)
            # Computes the new TAB temperature
            NewTAB = self.Tw + self.eta * deltaTsz
            # Tests if TAB > Tmand return zero
            if NewTAB > self.Tm:
                return 0, 0, 0, 0
            # Tests for the convergence of TAB (criterion is TAB_precision)
            if abs(NewTAB - TAB) <= TAB_precision:
                return NewTAB, Fs, deltaTsz, kAB
            # Affects the new TAB
            TAB = NewTAB
            maxLoops -= 1
        # Oups ! no convergence at least after so many loops
        return 0, 0, 0, 0

    def _compute_Tc(self, Ff, Vc, deltaTsz):
        """Computes the Tc temperature."""
        Tc_precision = 1e-3  # Precision on the evaluation of Tc
        Tc = self.Tw + deltaTsz
        nIters = 0
        # while True:
        while nIters < 1e7:
            # Computes Cp and K for temperature Tc
            Cp = self._cp_law(Tc)
            # K = KLaw(Tc) Not used here
            # Increment of temperature at the interface
            deltaTc = Ff * Vc / (self.mchip * Cp)
            # New interfacial temperature TC
            NewTc = self.Tw + deltaTsz + deltaTc
            # Tests for the convergence of Tc (criterion is Tc_precision)
            if abs(NewTc - Tc) <= Tc_precision:
                # print('%d iterations!' % nIters)
                return Tc, deltaTc
            # Affects the new Tc
            Tc = NewTc
            # Increments iterations
            nIters = nIters + 1
        # If can't calculate Tc, return None.
        return None

    def _compute_Toint_Kchip(self, C0, phi, delta):
        """
        Computes the Toint and Kchip values.
        """
        # Length of the first shear band lAB
        lAB = self.cutting_depth / math.sin(phi)
        # Speed along the first shear band
        Vs = self.cutting_speed * math.cos(self.cutting_angle) / math.cos(phi - self.cutting_angle)
        # Chip thickness
        t2 = self.cutting_depth * math.cos(phi - self.cutting_angle) / math.sin(phi)
        # Chip speed
        Vc = self.cutting_speed * math.sin(phi) / math.cos(phi - self.cutting_angle)
        # Plastic deformation in the AB zone
        gammaAB = 1 / 2 * math.cos(self.cutting_angle) / (math.sin(phi) * math.cos(phi - self.cutting_angle))
        EpsAB = gammaAB / math.sqrt(3)
        # Deformation rate in tge AB zone
        gammapAB = C0 * Vs / lAB  # not here
        EpspAB = gammapAB / math.sqrt(3)
        # Computes the TAB temperature
        TAB, Fs, deltaTsz, kAB = self._compute_AB(lAB, phi, Vs, EpsAB, EpspAB)
        # If TAB > self.Tw returns an error
        if TAB == 0:
            return 0, 1e10, 0, 0
        # Computes neq using Lalwani expression
        neq = self.jc_n * self.jc_B * EpsAB**self.jc_n / (self.jc_A + self.jc_B * EpsAB**self.jc_n)
        # Computes the theta angle
        theta = math.atan(1 + math.pi / 2 - 2 * phi - C0 * neq)
        # Computes the resultant force R depending on Fs and theta
        R = Fs / math.cos(theta)
        # Computes the lambda parameter
        Lambda = theta + self.cutting_angle - phi
        # Computes internal forces
        Ff = R * math.sin(Lambda)
        Fn = R * math.cos(Lambda)
        Fc = R * math.cos(theta - phi)
        Ft = R * math.sin(theta - phi)
        # Computes SigmaNp
        sigmaNmax = kAB * (1.0 + math.pi / 2.0 - 2.0 * self.cutting_angle - 2.0 * C0 * neq)
        # Tool/Chip contact length
        lc = (
            self.cutting_depth
            * math.sin(theta)
            / (math.cos(Lambda) * math.sin(phi))
            * (1 + C0 * neq / (3 * (1 + 2 * (math.pi / 4 - phi) - C0 * neq)))
        )
        # Stress along the interface
        Toint = Ff / (lc * self.cutting_width)
        # Equivalent deformation along the interface
        gammaM = lc / (delta * t2)
        gammaInt = 2 * gammaAB + gammaM / 2
        Epsint = gammaInt / math.sqrt(3)
        # Rate of deformation along the interface
        gammapInt = Vc / (delta * t2)
        Epspint = gammapInt / math.sqrt(3)
        # Contact temperature along the interface
        Tc, deltaTc = self._compute_Tc(Ff, Vc, deltaTsz)
        # K and Cp function of the Tc temperature
        K = self._k_law(Tc)
        Cp = self._cp_law(Tc)
        # Computes the RT factor
        RT = (self.rho * Cp * self.cutting_speed * self.cutting_depth) / K
        # Computes the delta self.Tm value
        # deltaTm = deltaTc*10.0**(0.06-0.195*delta*math.sqrt(RT*t2/lc)+0.5*math.log10(RT*t2/lc))
        deltaTm = deltaTc * 10 ** (0.06 - 0.195 * delta * math.sqrt(RT * t2 / lc)) * math.sqrt(RT * t2 / lc)
        # Mean temperature along the interface
        Tint = self.Tw + deltaTsz + self.psi * deltaTm
        # Stress flow within the chip
        kchip = (1 / math.sqrt(3)) * self._johnson_cook(Epsint, Epspint, Tint)
        # Computes the normal stress
        sigmaN = Fn / (lc * self.cutting_width)
        return Toint, kchip, sigmaN, sigmaNmax, Fc, Ft, t2

    def _reinitialize_para_opt1(self, paramsOpt1):
        """Initialization of the internal parameters for the optimization procedure."""
        paramsOpt1['delta'].value = (paramsOpt1['delta'].max + paramsOpt1['delta'].min) / 2

    def _reinitialize_para_opt2(self, paramsOpt2):
        paramsOpt2['C0'].value = (paramsOpt2['C0'].max + paramsOpt2['C0'].min) / 2
        paramsOpt2['phi'].value = (paramsOpt2['phi'].max + paramsOpt2['phi'].min) / 2

    def _internal_fitting_function(self, paramsOpt2, delta):
        """Internal fitting function on C0 and phi."""
        C0 = paramsOpt2['C0'].value
        phi = paramsOpt2['phi'].value
        # Computes the internal parameters
        Toint, kchip, sigmaN, sigmaNmax, Fc, Ft, t2 = self._compute_Toint_Kchip(C0, phi, delta)
        # Test if there was self.A bug in the last run
        if Toint == 0:
            print(self.cutting_speed * 60, self.cutting_depth, 180 / math.pi * phi, C0, delta, ' FAILED\self.n')
        # Computes the gap for the optimizer
        Gap = [(Toint - kchip), (sigmaN - sigmaNmax)]
        # Return the gap value
        return Gap

    def _fitting_function(self, paramsOpt1):
        """External fitting function on delta."""
        paramsOpt2 = lmfit.Parameters()
        paramsOpt2.add('C0', value=6, min=2, max=10)
        paramsOpt2.add('phi', value=26.5 * math.pi / 180, min=8 * math.pi / 180, max=45 * math.pi / 180)
        self._reinitialize_para_opt2(paramsOpt2)
        delta = paramsOpt1['delta'].value
        fitOpt1 = lmfit.minimize(self._internal_fitting_function, paramsOpt2, args=(delta,))
        paramsOpt2 = fitOpt1.params
        self.COF = paramsOpt2['C0'].value
        self.phiF = paramsOpt2['phi'].value
        Toint, kchip, sigmaN, sigmaNmax, Fc, Ft, t2 = self._compute_Toint_Kchip(
            paramsOpt2['C0'].value, paramsOpt2['phi'].value, delta
        )
        # Returns the cutting Force
        return [Fc]

    def _optimize(self):
        """Internal optimization procedure"""
        # Initialize the lists of parameters
        paramsOpt1 = lmfit.Parameters()
        # Initial values
        paramsOpt1.add('delta', value=0.015, min=0.005, max=0.2)
        self._reinitialize_para_opt1(paramsOpt1)
        # Calls the optimizer
        myfit = lmfit.minimize(self._fitting_function, paramsOpt1)
        # Get the results
        paramsOpt1 = myfit.params
        delta = paramsOpt1['delta'].value
        return delta


class ExtendedOxleyTask:
    """An extended Oxley task consists of finding the process parameters that optimize the process
    described by the extended Oxley simulator given the material parameters.

    Parameters
    ----------
    material_parameters: list
        A list with material parameters, or a list of lists with material parameters, in case of a task with multiple goals.
    total_depth: Union(float,list)
        Total depth to be removed from material, or a list of depths, in case of a task with multiple goals.
    total_length: Union(float,list)
        Total length to be removed, or a list of lengths, in case of a task with multiple goals.
    """

    def __init__(self, material_parameters, total_depth, total_length):
        self.material_parameters = material_parameters
        self.total_depth = total_depth
        self.total_length = total_length
        if not isinstance(self.total_depth, collections.abc.Sequence):
            self.extOxley = ExtendedOxleySimulator(material_parameters, total_depth, total_length)
        else:
            self.extOxley = [
                ExtendedOxleySimulator(self.material_parameters[idx], self.total_depth[idx], self.total_length[idx])
                for idx in range(len(self.material_parameters))
            ]

    def performance_measure(self, speed, Fc, Ft, n_layers, total_length):
        """Objectives to be minimized."""
        production_time = (total_length / speed) * n_layers
        tool_wear = (speed * np.exp(abs(Fc)) + 0.1 * speed * np.exp(abs(Ft))) * n_layers
        return [production_time, tool_wear, abs(Fc), abs(Ft)]

    def success_criterion(self, speed, Fc, Ft):
        """Returns True for all conditions if task-performing system provides a feasible solution"""

        if speed >= 50.0:
            SpeedFeas = False
        else:
            SpeedFeas = True
        if abs(Fc) >= 500.0 or abs(Ft) >= 500.0:
            ForceFeas = False
        else:
            ForceFeas = True

        return [SpeedFeas, ForceFeas]

    @property
    def params(self):
        return self.material_parameters

    def get_params(self):
        return self.params

    def run(self, tps, adapt=False, hypervolume_ref=float('inf')):
        tps.performance_measure = self.performance_measure
        tps.success_criterion = self.success_criterion
        tps.fit(self.extOxley, adapt=adapt, hypervolume_ref=hypervolume_ref)

        # pareto_outputs = tps.predict(self.extOxley)
        # print("Pareto Front obtained:")
        # for idx in range(len(pareto_outputs)):
        # shear_angle, Fc, Ft, tc, n_layers = pareto_outputs[idx]
        # speed = tps.pareto_front[idx]['solution'][0]
        # performance = self.performance_measure(speed, Fc, Ft, n_layers)
        # print("Solution=%s, Output=%s, Performance=%s" % (tps.pareto_front[idx]['solution'], pareto_outputs[idx], performance))


class TaskContextProofConcept:
    """Task context for the proof of concept."""

    def __init__(self, logger):
        self.params_names = [
            'Initial Temperature (T0)',
            'Ambient Temperature (Tw)',
            'Density (rho)',
            'eta',
            'psi',
            'A Coefficient (jc_A)',
            'B Coefficient (jc_B)',
            'n Coefficient (jc_n)',
            'C Coefficient (jc_C)',
            'm Coefficient (jc_m)',
            'Melting Temperature (Tm)',
            'Epsilon Dot Coefficient (jc_Epsp0)',
        ]

        self.tasks_params = {
            # "aluminium": [273.15,300.0,2700.0,0.9,0.9,1.67e+8,5.96e+8,0.551,0.001,0.859,893.0,1.0]},
            'steel': [273.15, 300.0, 7860.0, 0.9, 0.9, 7.92e8, 5.10e8, 0.26, 0.014, 1.03, 1790.0, 1.0],
            'tungsten_alloy': [273.15, 300.0, 17600.0, 0.9, 0.9, 1.51e9, 1.77e8, 0.12, 0.016, 1.0, 1723.0, 1.0],
            'steel_dummy': [273.15, 300.0, 7860.0, 0.9, 0.9, 5.82e8, 4.65e8, 0.325, 0.008, 1.3, 1790.0, 1.0],
            'inconel_718': [273.15, 300.0, 8242.0, 0.9, 0.9, 9.28e8, 9.79e8, 0.245847, 0.0056, 1.80073, 1623.15, 0.001],
        }

        self.total_depth = 1.0
        self.total_length = 1.0

        task_context_desc = (
            'Training on different materials for comparison of Pareto fronts obtained.\nMaterial Parameters:\n%s\n \
             Tasks:\n%s\ntotal_depth=%f\ntotal_length=%f\nNumber of repetitions defined by user.'
            % (self.params_names, self.tasks_params, self.total_depth, self.total_length)
        )

        self.logger = logger
        self.logger.register_task_context(task_context_desc)

    def run(self, tps, n_runs):
        for material in self.tasks_params.keys():
            material_params = self.tasks_params[material]
            task = ExtendedOxleyTask(material_params, self.total_depth, self.total_length)

            for i in range(n_runs):
                print('%s run %d' % (material, i + 1))
                task.run(tps)
                tps.register_data(material, 'training')

        self.logger.save(tps.save_solutions)


class TaskContextAdaption:
    """Task context for searching from scratch and adapting for source and target task pairs.

    Parameters:
    -----------
    logger: Logger
        Logger object for the experiment.
    params_names: list
        List of names for the material parameters.
    tasks_params: dict
        Dictionary where each key is a material name and its corresponding value is the list of material parameters.
    total_depth: float
        Total depth of material to be cut (constant accross different materials).
    total_length: float
        Total length of material to be cut (constant accross different materials).
    tasks_training: list
        List of tasks for training (material names).
        These are tasks that the task-performing system will optimize from scratch.
        Elements are tuples, as one can optimize for one or more tasks at the same time (varying goals).
    tasks_adaption: list
        List of tasks for adaption (source and target material names).
        Elements are tuples, where the first value is the source task, and the second is the target task.
        Task-performing system will adapt solutions from source to target.
    path_source: str
        Path from where solutions for adaption should be loaded from. Default is current log directory.
    path_target: str
        Path from where reference values for adaption should be loaded from. Default is current log directory.
    """

    def __init__(
        self,
        logger,
        params_names,
        tasks_params,
        total_depth,
        total_length,
        tasks_training,
        tasks_adaption,
        path_source='',
        path_target='',
    ):
        self.logger = logger
        self.params_names = params_names
        self.tasks_params = tasks_params
        self.total_depth = total_depth
        self.total_length = total_length
        self.tasks_training = tasks_training
        self.tasks_adaption = tasks_adaption

        if path_source == '':
            self.path_source = self.logger.log_path
        else:
            self.path_source = path_source
        if path_target == '':
            self.path_target = self.logger.log_path
        else:
            self.path_target = path_target

        task_context_desc = (
            'Search from scratch followed by adaption for different task pairs.\n'
            'Material Parameters:\n%s\nTasks Parameters:\n%s\nTotal Depth=%f\nTotal Length=%f\n'
            'Tasks Training:\n%s\nTasks Adaption:\n%s\nPath Source:\n%s\nPath Target:\n%s\n'
            'Number of repetitions defined by user.'
            % (
                self.params_names,
                self.tasks_params,
                self.total_depth,
                self.total_length,
                self.tasks_training,
                self.tasks_adaption,
                self.path_source,
                self.path_target,
            )
        )

        self.logger.register_task_context(task_context_desc)

    def run(self, tps, n_runs):
        # First, run tasks from scratch.
        for tasks in self.tasks_training:
            task_name = '_and_'.join(tasks)
            material_params = [self.tasks_params[material] for material in tasks]
            total_depth = [self.total_depth] * len(tasks)
            total_length = [self.total_length] * len(tasks)
            task = ExtendedOxleyTask(material_params, total_depth, total_length)
            for rep in range(n_runs):
                print('%s run %d.' % (task_name, rep + 1))
                task.run(tps)
                tps.register_data(task_name, 'training')
                self.logger.save(tps.save_solutions)
                self.logger.reset_data()

        # Then, run pairs of source x target.
        for source, target in self.tasks_adaption:
            task_name_source = '_and_'.join(source)
            task_name = '%s_to_%s' % (task_name_source, target)
            task = ExtendedOxleyTask(self.tasks_params[target], self.total_depth, self.total_length)
            # For now, loads hypervolume as reference for target.
            # If different measures are used, one option is to pass path_target to TPS and let it handle the
            # loading of the reference values internally.
            # Another option is to name the file for cost with a standard name (for example, cost.csv).
            with open(self.path_target + target + '/training/hypervolume.csv', 'r', newline='') as f:
                reader = csv.reader(f, delimiter=',')
                hypervolumes_ref = [max([float(x) for x in row]) for row in reader]
            for rep in range(n_runs):
                print('%s run %s' % (task_name, rep + 1))
                adapt_path = (
                    self.path_source + task_name_source + '/training/solutions/solutions_repetition_' + str(rep + 1)
                )
                task.run(tps, adapt=adapt_path, hypervolume_ref=hypervolumes_ref[rep])
                tps.register_data(task_name, 'adaption')
                self.logger.save(tps.save_solutions)
                self.logger.reset_data()


class TaskContextStudyParams:
    """Task context for doing a study of parameters for a given TPS.

    Parameters:
    -----------
    logger: Logger
        Logger object for the experiment.
    params_names: list
        List of names for the material parameters.
    tasks_params: dict
        Dictionary where each key is a material name and its corresponding value is the list of material parameters.
    total_depth: float
        Total depth of material to be cut (constant accross different materials).
    total_length: float
        Total length of material to be cut (constant accross different materials).
    tasks_training: list
        List of tasks for training (material names).
        These are tasks that the task-performing system will optimize from scratch.
        Elements are tuples, as one can optimize for one or more tasks at the same time (varying goals).
    tasks_adaption: list
        List of tasks for adaption (source and target material names).
        Elements are tuples, where the first value is the source task, and the second is the target task.
        Task-performing system will adapt solutions from source to target.
    path_source: str
        Path from where solutions for adaption should be loaded from. Default is current log directory.
    path_target: str
        Path from where reference values for adaption should be loaded from. Default is current log directory.
    varied_params_names = list
        List with the names of the parameters being varied.
    varied_params_values = dict
        Dictionary where each key is a parameter name and the corresponding value is a list of values the
        parameter can assume.
    include_target = bool
        True if target in adaption phase is also included in the parameter study.
    """

    def __init__(
        self,
        logger,
        params_names,
        tasks_params,
        total_depth,
        total_length,
        tasks_training,
        tasks_adaption,
        varied_params_names,
        varied_params_values,
        include_target,
        path_source='',
        path_target='',
    ):
        self.logger = logger
        self.params_names = params_names
        self.tasks_params = tasks_params
        self.total_depth = total_depth
        self.total_length = total_length
        self.tasks_training = tasks_training
        self.tasks_adaption = tasks_adaption
        self.varied_params_names = varied_params_names
        self.varied_params_values = varied_params_values
        self.include_target = include_target

        if path_source == '':
            self.path_source = self.logger.log_path
        else:
            self.path_source = path_source
        if path_target == '':
            self.path_target = self.logger.log_path
        else:
            self.path_target = path_target

        task_context_desc = (
            'Study of parameters.\n'
            'Material Parameters:\n%s\nTasks Parameters:\n%s\nTotal Depth=%f\nTotal Length=%f\n'
            'Tasks Training:\n%s\nTasks Adaption:\n%s\nPath Source:\n%s\nPath Target:\n%s\n'
            'Varied Params Names:\n%s\nVaried Params Values:\n%s\nInclude Target:%s\n'
            'Number of repetitions defined by user.'
            % (
                self.params_names,
                self.tasks_params,
                self.total_depth,
                self.total_length,
                self.tasks_training,
                self.tasks_adaption,
                self.path_source,
                self.path_target,
                self.varied_params_names,
                self.varied_params_values,
                self.include_target,
            )
        )

        self.logger.register_task_context(task_context_desc)

    def run(self, tps, n_runs):
        # Product of parameter values.
        for values in itertools.product(*[self.varied_params_values[key] for key in self.varied_params_names]):
            # Set new values in TPS.
            for param, x in zip(self.varied_params_names, values):
                tps.__setattr__(param, x)

            # Base for task names.
            # Currently only working for integer values.
            base = '_'.join(['%s_%d' % (param, x) for (param, x) in zip(self.varied_params_names, values)])

            # Run tasks from scratch.
            for tasks in self.tasks_training:
                task_name = base + '_' + '_and_'.join(tasks)
                material_params = [self.tasks_params[material] for material in tasks]
                total_depth = [self.total_depth] * len(tasks)
                total_length = [self.total_length] * len(tasks)
                task = ExtendedOxleyTask(material_params, total_depth, total_length)
                for rep in range(n_runs):
                    print('%s run %s' % (task_name, rep + 1))
                    task.run(tps)
                    tps.register_data(task_name, 'training')
                    self.logger.save(tps.save_solutions)
                    self.logger.reset_data()

            # Run pairs of source x target.
            for source, target in self.tasks_adaption:
                task_name = base + '_' + '_and_'.join(source) + '_to_' + target
                task_name_source = base + '_' + '_and_'.join(source)
                if self.include_target:
                    task_name_target = base + '_' + target
                else:
                    task_name_target = target
                task = ExtendedOxleyTask(self.tasks_params[target], self.total_depth, self.total_length)
                with open(self.path_target + task_name_target + '/training/hypervolume.csv', 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=',')
                    hypervolumes_ref = [max([float(x) for x in row]) for row in reader]
                for rep in range(n_runs):
                    print('%s run %s' % (task_name, rep + 1))
                    adapt_path = (
                        self.path_source + task_name_source + '/training/solutions/solutions_repetition_' + str(rep + 1)
                    )
                    task.run(tps, adapt=adapt_path, hypervolume_ref=hypervolumes_ref[rep])
                    tps.register_data(task_name, 'adaption')
                    self.logger.save(tps.save_solutions)
                    self.logger.reset_data()


class OxleyTaskEstimator(ABC):
    """An abstract TPS (Task Performing System) class defining the minimal interface of a task-performing system required."""

    @abstractmethod
    def __init__(self, logger, n_process_params):
        """Create a new instance.

        Args:
            logger: Logger
                Logger object for all experimental information and data.
            n_process_params: int
                Number of process parameters to be optimized.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, extOxley, adapt=False, hypervolume_ref=float('inf')):
        """Optimize the process parameters for the Oxley simulator.

        Args:
            extOxley: ExtendedOxleySimulator
                An instance of an Oxley simulator.
            adapt: Union(bool, str)
                False if running from scratch, True if using solution from previous run, str if loading solution from somewhere.
            hypervolume_ref: float
                Inf if searching from scratch.
                If adapting, equals to the hypervolume previously found for the target when searching from scratch.
                Provides a reference for when to stop searching.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, extOxley):
        """Evaluate learned solutions from Pareto front.

        Args:
            extOxley: ExtendedOxleySimulator
                An instance of an Oxley simulator.

        Returns:
            y: array of dimensions (n_solutions, n_outputs).
                The computed outputs for each solution in the Pareto front.
        """
        raise NotImplementedError

    @abstractmethod
    def register_data(self, task, exp_phase):
        """Stores data from run in the logger object.

        Args:
            task: str
                Task identification for logger.
            exp_phase: str
                Identification of type of run for logger ("training" or "adaption").
        """
        raise NotImplementedError

    @abstractmethod
    def save_solutions(self, solutions, path):
        """Save a sequence of solutions to a path.

        Args:
            solutions: Sequence
                A sequence of solutions.
            path: str
                Path where to save the solutions.
        """
        raise NotImplementedError

    @abstractmethod
    def load_solutions(self, path):
        """Loads solutions from a path.

        Args:
            path: str
                Path from where to load solutions.

        Returns:
            solutions: Sequence
                A sequence of loaded solutions.
        """
        raise NotImplementedError

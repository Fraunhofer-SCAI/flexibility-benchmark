import random, json, itertools, os
import numpy as np

from multiprocessing import Pool
from functools import reduce

from flexbench.core import Logger
from flexbench.exceptions import NotFittedError
from flexbench.benchmarks.oxley_model_iwm import ExtendedOxleyTask, TaskContextProofConcept, OxleyTaskEstimator


class RandomSamplingEstimator(OxleyTaskEstimator):
    """Estimation of the Pareto Front via random sampling.

    Parameters
    ----------
    logger: Logger
        Logger object for all experimental information and data.
    n_process_params: int
        Number of process parameters to be optimized.
    range_process_params: array of dimensions [n_process_params, 2]
        The range from witch to sample for each process parameter.
    n_samples: int
        Number of solutions to be sampled.

    Attributes
    ----------
    pareto_front: list[dict]
        Estimated Pareto Front (solution, output, feasibility, performance).
    nEvals: int
        The number of evaluations that were performed to estimate the Pareto Front (constant given n_samples).
    """

    def __init__(self, logger, n_process_params, range_process_params, n_samples, n_parallel, verbosity):
        self.n_process_params = n_process_params
        self.range_process_params = range_process_params
        self.n_samples = n_samples
        self.n_parallel = n_parallel
        self.verbosity = verbosity

        system_desc = (
            'RandomSamplingEstimator\nn_process_params=%d\nrange_process_params=%s\nn_samples=%d\nn_parallel=%d\nverbosity=%s'
            % (
                self.n_process_params,
                np.array(self.range_process_params),
                self.n_samples,
                self.n_parallel,
                self.verbosity,
            )
        )

        self.logger = logger
        self.logger.register_system(system_desc)

    def _find_pareto_front(self, samples):
        # Select only feasible solutions.
        samples = [x for x in samples if np.all(x['feasibility'])]

        if self.verbosity:
            print('Found %d feasible solutions.' % len(samples))

        # Extract Pareto Front.
        pareto_front = []
        for s1 in samples:
            if np.all(
                [
                    (np.any(s1['performance'] < s2['performance']) or np.all(s1['performance'] == s2['performance']))
                    for s2 in samples
                ]
            ):
                pareto_front.append(s1)

        if self.verbosity:
            print('%d solutions in the Pareto front.' % len(pareto_front))

        # Return Pareto Front.
        return pareto_front

    def _sample_n(self, arg):
        n = arg[0]
        extOxley = arg[1]
        samples = []

        for i in range(n):
            # Sample random solution.
            solution = [
                random.uniform(self.range_process_params[idx][0], self.range_process_params[idx][1])
                for idx in range(self.n_process_params)
            ]

            # Calculate output via simulation.
            try:
                output = extOxley.step(solution)
            except (ValueError, TypeError):
                output = [float('Inf')] * 5

            # Check feasibility using as arguments: cutting_speed, Fc, Ft.
            feasibility = self.success_criterion(solution[0], output[1], output[2])

            # Calculate performance using as arguments: cutting_speed, Fc, Ft, n_layers.
            performance = np.array(self.performance_measure(solution[0], output[1], output[2], output[4]))

            samples.append(
                {'solution': solution, 'output': output, 'feasibility': feasibility, 'performance': performance}
            )

        return samples

    def fit(self, extOxley, adapt=False):
        intermediate_n = int(self.n_samples / self.n_parallel)
        rest_n = self.n_samples % self.n_parallel
        vector_n = np.array([intermediate_n] * self.n_parallel) + np.array(
            ([1] * rest_n) + ([0] * (self.n_parallel - rest_n))
        )

        if self.verbosity:
            print('Sampling...')
            print('Number of samplings per parallel process: %s' % vector_n)

        pool = Pool(self.n_parallel)
        samples = pool.map(self._sample_n, [[x, extOxley] for x in vector_n])
        samples = reduce(lambda x, y: x + y, samples)
        pool.close()

        self.pareto_front = self._find_pareto_front(samples)
        self.nEvals = self.n_samples

    def predict(self, extOxley):
        if not hasattr(self, 'pareto_front'):
            raise NotFittedError('This RandomSamplingEstimator instance has not been fitted yet.')

        return [extOxley.step(x['solution'][: self.n_process_params]) for x in self.pareto_front]

    def register_data(self, task, exp_phase):
        self.logger.register_data(task, exp_phase, 'cost', [self.nEvals])
        self.logger.register_data(task, exp_phase, 'solution', self.pareto_front)

    def save_solutions(self, solutions, path):
        for x in solutions:
            x['performance'] = list(x['performance'])

        with open(path + '.json', 'w') as f:
            json.dump(solutions, f)

    def load_solutions(self, path):
        pass


if __name__ == '__main__':
    """Proof of concept: Perform a random sampling on some tasks to compare the Pareto fronts."""

    # Initialize logger, with log path and experiment name.
    exp_name = '1-1_proof_of_concept_random_sampling'
    log_path = '../../flexbench-data/'
    logger = Logger(exp_name=exp_name, log_path=log_path)

    # Initilize task context.
    task_context = TaskContextProofConcept(logger)

    # Initialize task-performing system.
    # tps = RandomSamplingEstimator(
    # logger=logger,
    # n_process_params=3,
    # range_process_params=[
    # [0.1,5.0],                 # Range for cutting speed.
    # [-0.5,1.0],                # Range for cutting angle.
    # [1.0e-6,1.0e-3]            # Range for cutting depth.
    # ],
    # n_samples=1000,
    # n_parallel = 4,
    # verbosity = False
    # )

    # Call run method from task context.
    n_runs = 10
    # task_context.run(tps, n_runs=n_runs)

    #####
    # Evaluation of Pareto front obained for one material on other materials.
    #####

    # Create directory.
    path_save_pre = '%s%s/pareto_fronts_evaluation/' % (log_path, exp_name)

    try:
        os.mkdir(path_save_pre)
    except FileExistsError:
        pass

    # Pairs of materials (source and target).
    materials = ['steel', 'tungsten_alloy', 'steel_dummy', 'inconel_718']
    pairs = itertools.permutations(materials, 2)

    for pair in pairs:
        source = pair[0]
        target = pair[1]

        print('Evaluating Pareto front from %s on %s...' % (source, target))

        # Create directory.
        path_save = '%spareto_from_%s_evaluated_on_%s/' % (path_save_pre, source, target)

        try:
            os.mkdir(path_save)
        except FileExistsError:
            pass

        for rep in range(1, n_runs + 1):
            print('Repetition %d' % rep)

            # Load source Pareto front.
            with open(
                '%s%s/%s/training/solutions/solutions_repetition_%d.json' % (log_path, exp_name, source, rep), 'r'
            ) as f:
                pareto_front = json.load(f)

            # Evaluate source Pareto front on target material.
            task = ExtendedOxleyTask(
                task_context.tasks_params[target], task_context.total_depth, task_context.total_length
            )

            for solution in pareto_front:
                try:
                    output = task.extOxley.step(solution['solution'])
                except (ValueError, TypeError):
                    output = [float('Inf')] * 5

                feasibility = task.success_criterion(solution['solution'][0], output[1], output[2])
                performance = task.performance_measure(solution['solution'][0], output[1], output[2], output[4])

                solution['output'] = output
                solution['feasibility'] = feasibility
                solution['performance'] = performance

            # Save result.
            with open('%ssolutions_%d.json' % (path_save, rep), 'w') as f:
                json.dump(pareto_front, f)

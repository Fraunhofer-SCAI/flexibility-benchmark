import random, json, itertools, os, collections
import numpy as np

from multiprocessing import Pool
from functools import reduce

from deap import base, creator, tools, algorithms
from pymoo.indicators.hv import HV

from flexbench.core import Logger
from flexbench.exceptions import NotFittedError
from flexbench.benchmarks.oxley_model_iwm import (
    ExtendedOxleyTask,
    TaskContextAdaption,
    TaskContextStudyParams,
    OxleyTaskEstimator,
)


def _calc_fitness(ind, extOxley, success_criterion, performance_measure):
    # Calculate output via simulation.
    try:
        output = extOxley.step(ind.decoded)
    except (ValueError, TypeError):
        output = [float('Inf')] * 5

    # Check feasibility using as arguments: cutting_speed, Fc, Ft.
    feasibility = success_criterion(ind.decoded[0], output[1], output[2])

    # Calculate performance using as arguments: cutting_speed, Fc, Ft, n_layers, total_length.
    performance = np.array(performance_measure(ind.decoded[0], output[1], output[2], output[4], extOxley.total_length))

    return output, feasibility, performance


class NSGAIIEstimator(OxleyTaskEstimator):
    """Estimation of the Pareto Front via NSGA-II.

    Parameters
    ----------
    logger: Logger
        Logger object for all experimental information and data.
    n_process_params: int
        Number of process parameters to be optimized.
    range_process_params: array of dimensions [n_process_params, 2]
        The range from witch to sample process params for initial population.
    range_objectives: array of dimensions [n_objectives, 2]
        Possible ranges for objectives, for normalization.
    pop_size: int
         Population size.
    n_gens: int
        Number of generations. Runs for total number when searching from scratch.
    use_tourn: bool
        Tells if tournament should be used instead of varAnd from DEAP.
    tourn_size: int
        Tournament size for selection. Selects based on non-domination rank and crowding distance.
    eta_cross: float
        Eta parameter for simulated binary bounded crossover.
    eta_mut: float
        Eta parameter for polynomial bounded mutation.
    load_percentage: float
        The percentage [0,1] of the initial population that should be loaded from a source.
        Used only when adapting.
    target_percentage: float
        The percentage [0,1] of the hypervolume target that should be achieved.
    epoch_length: int
        Number of generations that define an epoch.
        Used when optimizing for multiple goals (when passing a vector of 'extOxley' to 'fit()').
        Changes the goal each 'epoch_length' generations.
        Currently, multiple goals do not work for adaption, only for optimizing a source population without providing
        a target hypervolume.
    gene_length: int
        Number of positions in a gene. Only one position is active at a time.
        A different position may be activated by mutation.
    n_parallel: int
        Number of parallel fitness evaluations.
    verbose: bool
        Whether to display more information throughout run.

    Attributes
    ----------
    pareto_front: list[dict]
        Pareto Front (solution, output, feasibility, performance) with best hypervolume found in run.
        When using multiple goals, updated when the value for the current goal increases.
        Currently, we do not keep a best Pareto front for each goal.
    hypervolume: list[float]
        List with best hypervolume found in run.
        When using multiple goals, contains the hypervolume of the same Pareto front for each goal.
        May be different than the hypervolume for each goal of the current Pareto front stored.
    nEvals: int
        The number of evaluations that were performed to estimate the Pareto Front.
        When searching from scratch, it corresponds to the generation with the last improvement in hypervolume.
        When adapting, it correponds to the generation where the hypervolume matches the target reference.
    """

    def __init__(
        self,
        logger,
        n_process_params,
        range_process_params,
        range_objectives,
        pop_size=100,
        n_gens=50,
        use_tourn=False,
        tourn_size=2,
        eta_cross=30,
        eta_mut=20,
        load_percentage=1.0,
        target_percentage=1.0,
        epoch_length=50,
        gene_length=1,
        n_parallel=1,
        verbosity=True,
    ):
        self.n_process_params = n_process_params
        self.range_process_params = range_process_params
        self.range_objectives = range_objectives
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.use_tourn = use_tourn
        self.tourn_size = tourn_size
        self.eta_cross = eta_cross
        self.eta_mut = eta_mut
        self.load_percentage = load_percentage
        self.target_percentage = target_percentage
        self.epoch_length = epoch_length
        self.gene_length = gene_length
        self.n_parallel = n_parallel
        self.verbosity = verbosity

        system_desc = (
            'NSGAIIEstimator\nn_process_params=%d\nrange_process_params=%s\nrange_objectives=%s\npop_size=%d\n'
            'n_gens=%d\nuse_tourn=%s\ntourn_size=%d\neta_cross=%f\neta_mut=%f\nload_percentage=%f\n'
            'target_percentage=%f\nepoch_length=%d\ngene_length=%d\nn_parallel=%d\nverbosity=%s'
            % (
                self.n_process_params,
                np.array(self.range_process_params),
                np.array(self.range_objectives),
                self.pop_size,
                self.n_gens,
                self.use_tourn,
                self.tourn_size,
                self.eta_cross,
                self.eta_mut,
                self.load_percentage,
                self.target_percentage,
                self.epoch_length,
                self.gene_length,
                self.n_parallel,
                self.verbosity,
            )
        )

        self.logger = logger
        self.logger.register_system(system_desc)

    def _deap_setup(self):
        """Sets up classes and functions used by DEAP."""

        # Initialize deap toolbox.
        self.toolbox = base.Toolbox()

        # Create FitnessMin class for multiple (four) objectives.
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,) * 4)

        # Create Individual class, that is a list with fitness class FitnessMin.
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # Individual (list of float values).
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self._create_ind)

        # Population (list of individuals of arbitrary length).
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Fitness function.
        self.toolbox.register('evaluate', _calc_fitness)

        # Selection mechanism.
        self.toolbox.register('select', tools.selNSGA2)

        # Crossover
        self.toolbox.register('mate', self._mate)

        # Mutation
        self.toolbox.register('mutate', self._mutate)

    def _create_ind(self):
        ind = []
        for idx in range(self.n_process_params):
            ind.append(random.randint(1, self.gene_length))  # First integer indicates which position is active.
            ind.extend(
                [
                    random.uniform(self.range_process_params[idx][0], self.range_process_params[idx][1])
                    for i in range(self.gene_length)
                ]
            )
        return ind

    def _decode(self, ind):
        """Returns active positions from each gene as process parameters."""
        return [ind[i + ind[i]] for i in range(0, len(ind), self.gene_length + 1)]

    def _encode(self, ind, decoded):
        """Replaces active positions from 'ind' with values provided in 'decoded' in place and returns result."""
        for i in range(len(decoded)):
            idx_gene = i * (self.gene_length + 1)
            ind[idx_gene + ind[idx_gene]] = decoded[i]
        return ind

    def _mate(self, parent1, parent2):
        """
        Applies crossover only to active positions.
        Mates in place and returns offspring to follow DEAP interface.
        """
        parent1.decoded, parent2.decoded = tools.cxSimulatedBinaryBounded(
            ind1=parent1.decoded,
            ind2=parent2.decoded,
            eta=self.eta_cross,
            low=[x[0] for x in self.range_process_params],
            up=[x[1] for x in self.range_process_params],
        )

        return self._encode(parent1, parent1.decoded), self._encode(parent2, parent2.decoded)

    def _mutate(self, ind):
        """
        Deactivates or activates positions and then applies mutation only to active ones.
        Mutates in place and returns offspring to follow DEAP interface.
        """
        # Activate or deactivate positions.
        for i in range(0, len(ind), self.gene_length + 1):
            if (random.random() < 1.0 / self.n_process_params) and self.gene_length > 1:
                current = ind[i]
                while ind[i] == current:
                    ind[i] = random.randint(1, self.gene_length)
        ind.decoded = self._decode(ind)

        # Mutate active positions.
        (ind.decoded,) = tools.mutPolynomialBounded(
            individual=ind.decoded,
            eta=self.eta_mut,
            low=[x[0] for x in self.range_process_params],
            up=[x[1] for x in self.range_process_params],
            indpb=1.0 / self.n_process_params,
        )

        return (self._encode(ind, ind.decoded),)

    def _find_pareto_front(self, pop):
        # Select only feasible solutions.
        pop_feasible = [x for x in pop if np.all(x.feasibility)]

        if self.verbosity:
            print('  Found %d feasible solutions.' % len(pop_feasible))

        # Extract Pareto Front.
        pareto_front = []
        for s1 in pop_feasible:
            if np.all(
                [
                    (
                        np.any(np.array(s1.fitness.values) < np.array(s2.fitness.values))
                        or np.all(np.array(s1.fitness.values) == np.array(s2.fitness.values))
                    )
                    for s2 in pop_feasible
                ]
            ):
                pareto_front.append(s1)

        if self.verbosity:
            print('  %d solutions in the Pareto front.' % len(pareto_front))

        # Return Pareto Front.
        return pareto_front

    def _calc_hypervolume(self, pareto_front):
        # Max and min values before and after normalization.
        max_value = 1.0
        min_value = 0.0

        min_data = np.array([float(x[0]) for x in self.range_objectives])
        max_data = np.array([float(x[1]) for x in self.range_objectives])

        # Extract objective values from current Pareto front.
        data = np.array([ind.fitness.values for ind in pareto_front])

        if len(data) == 0:
            return 0.0

        # Apply logarithmic scale to two first objectives.
        for i in range(2):
            data[:, i] = np.log(data[:, i])
            min_data[i] = np.log(min_data[i])
            max_data[i] = np.log(max_data[i])

        # Normalize.
        for i in range(len(data[0])):
            diff = max_value - min_value
            diff_data = max_data[i] - min_data[i]

            if diff_data == 0.0:
                diff_data = 1e-5

            data[:, i] = (((data[:, i] - min_data[i]) * diff) / diff_data) + min_value

        # Calculate hypervolume.
        ref_point = [max_value] * len(data[0])
        function_hv = HV(ref_point=ref_point)
        hypervolume = function_hv(data)

        return hypervolume

    def _tournament(self, pop, tourn_size):
        ordered_pop = self.toolbox.select(pop, len(pop))
        idx = min(random.sample(range(len(pop)), tourn_size))
        return ordered_pop[idx]

    def fit(self, extOxley, adapt=False, hypervolume_ref=float('inf')):
        # Set up DEAP.
        self._deap_setup()

        # Initialize population.
        if not adapt:
            pop = self.toolbox.population(n=self.pop_size)
        else:
            pop = self.load_solutions(adapt)
            pop_size_load = int(self.pop_size * self.load_percentage)
            pop = pop[:pop_size_load]
            pop.extend(self.toolbox.population(n=(self.pop_size - pop_size_load)))

        # Decode population.
        for ind in pop:
            ind.decoded = self._decode(ind)

        print('Start of evolution.')

        # Create pool for evaluation.
        pool = Pool(self.n_parallel)

        # Check if is optimizing for multiple goals to select current goal.
        extOxley_idx = 0
        if isinstance(extOxley, collections.abc.Sequence):
            n_goals = len(extOxley)
            extOxley_current = extOxley[extOxley_idx]
        else:
            n_goals = 1
            extOxley_current = extOxley

        # Evaluate population.
        results = pool.starmap(
            self.toolbox.evaluate,
            zip(
                pop,
                [extOxley_current] * self.pop_size,
                [self.success_criterion] * self.pop_size,
                [self.performance_measure] * self.pop_size,
            ),
        )
        for i in range(self.pop_size):
            pop[i].output = results[i][0]  # Return value 0 from evaluate: output.
            pop[i].feasibility = results[i][1]  # Return value 1 from evaluate: feasibility.
            pop[i].fitness.values = results[i][2]  # Return value 2 from evaluate: performance.

        # Update number of evaluations.
        if self.verbosity:
            print('  Evaluated %i individuals' % len(pop))
        self.nEvals = len(pop)
        nEvalsTotal = self.nEvals

        # Extract Pareto front and calculate hypervolume.
        self.hypervolume = [float('-Inf') for _ in range(n_goals)]
        self.pareto_front = self._find_pareto_front(pop)
        self.hypervolume[extOxley_idx] = self._calc_hypervolume(self.pareto_front)

        # Keep track of best population and hypervolume optimization.
        self.best_pop = pop
        self.vec_hypervolumes = [self.hypervolume[extOxley_idx]]

        # Print hypervolume.
        if self.verbosity:
            print('  Hypervolume = %s' % self.hypervolume)

        # Evolutionary loop.
        g = 0
        while g < self.n_gens and self.hypervolume[extOxley_idx] < (hypervolume_ref * self.target_percentage):
            g = g + 1
            if self.verbosity:
                print('-- Generation %i --' % g)

            # Generate offspring of population size by varying individuals.
            if not self.use_tourn:
                offspring = algorithms.varAnd(pop, self.toolbox, 1.0, 1.0)

            # Selection via tournament using NSGA-II ranking.
            else:
                offspring = []
                while len(offspring) < self.pop_size:
                    off1 = self.toolbox.clone(self._tournament(pop, self.tourn_size))
                    off2 = self.toolbox.clone(self._tournament(pop, self.tourn_size))
                    self.toolbox.mate(off1, off2)
                    self.toolbox.mutate(off1)
                    self.toolbox.mutate(off2)
                    offspring.extend([off1, off2])
                offspring = offspring[: self.pop_size]

            # Check if is optimizing for multiple goals to select current goal.
            if isinstance(extOxley, collections.abc.Sequence):
                extOxley_idx = int(np.floor((g - 1) / self.epoch_length)) % len(extOxley)
                extOxley_current = extOxley[extOxley_idx]

            # Evaluate offspring.
            results = pool.starmap(
                self.toolbox.evaluate,
                zip(
                    offspring,
                    [extOxley_current] * self.pop_size,
                    [self.success_criterion] * self.pop_size,
                    [self.performance_measure] * self.pop_size,
                ),
            )
            for i in range(len(offspring)):
                offspring[i].output = results[i][0]  # Return value 0 from evaluate: output.
                offspring[i].feasibility = results[i][1]  # Return value 1 from evaluate: feasibility.
                offspring[i].fitness.values = results[i][2]  # Return value 2 from evaluate: performance.

            # Keep track of total number of evaluations.
            if self.verbosity:
                print('  Evaluated %i individuals' % len(offspring))
            nEvalsTotal = nEvalsTotal + len(offspring)

            # Select individuals for new population.
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Calculate new Pareto front and hypervolume.
            pareto_front_new = self._find_pareto_front(pop)
            hypervolume_new = self._calc_hypervolume(pareto_front_new)

            # Add new hypervolume to keep track of optimization
            self.vec_hypervolumes.append(hypervolume_new)

            # Update data if hypervolume for current goal improved.
            # An option is to update best_pop even if hypervolume is the same to have a more optimized stored population
            # in a broader sense. As in some cases one might want to stop the search after reaching a reference value,
            # we do not do it here.
            if hypervolume_new > self.hypervolume[extOxley_idx]:
                self.nEvals = nEvalsTotal
                self.pareto_front = pareto_front_new
                self.hypervolume[extOxley_idx] = hypervolume_new
                self.best_pop = pop

            # Print hypervolume.
            if self.verbosity:
                print('  Best hypervolume found = %s' % self.hypervolume)
                print('  Current hypervolume = %f' % hypervolume_new)
                print('  nEvalsTotal = %d' % nEvalsTotal)
                print('  nEvals = %d' % self.nEvals)

        print('-- End of evolution --')

        # Close pool for evaluation.
        pool.close()

        # Adaption considers reference hypervolume to measure success.
        if adapt and self.hypervolume[extOxley_idx] < (hypervolume_ref * self.target_percentage):
            self.nEvals = -1

    def predict(self, extOxley):
        if not hasattr(self, 'pareto_front'):
            raise NotFittedError('This NSGAIIEstimator instance has not been fitted yet.')

        return [extOxley.step(x.decoded) for x in self.pareto_front]

    def register_data(self, task, exp_phase):
        # At the moment, saving best population found.
        # An option would be to save best Pareto front and add random reamining individuals from last population.
        # Or saving only best Pareto front and complete with random individuals when initializing.
        # As Pareto front is generally the entire population, this doesn't make sense here.
        self.logger.register_data(task, exp_phase, 'cost', [self.nEvals])
        self.logger.register_data(task, exp_phase, 'solution', self.best_pop)
        self.logger.register_data(task, exp_phase, 'hypervolume', self.vec_hypervolumes)

    def save_solutions(self, solutions, path):
        # Put in structure.
        solutions = [
            {
                'solution': x,
                'decoded': x.decoded,
                'output': x.output,
                'feasibility': x.feasibility,
                'performance': list(x.fitness.values),
            }
            for x in solutions
        ]

        # Save in file.
        with open(path + '.json', 'w') as f:
            json.dump(solutions, f)

    def load_solutions(self, path):
        # Load from file.
        with open(path + '.json', 'r') as f:
            solutions = json.load(f)

        # Return as individual objects used by DEAP.
        return [creator.Individual(x['solution']) for x in solutions]


if __name__ == '__main__':
    """Baseline adaption: Uses NSGA-II to calculate solutions from scratch and then adapt them for different pairs of source and target tasks."""

    # Base parameters.
    params_names = [
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

    tasks_params = {
        'steel': [273.15, 300.0, 7860.0, 0.9, 0.9, 7.92e8, 5.10e8, 0.26, 0.014, 1.03, 1790.0, 1.0],
        'tungsten_alloy': [273.15, 300.0, 17600.0, 0.9, 0.9, 1.51e9, 1.77e8, 0.12, 0.016, 1.0, 1723.0, 1.0],
        'steel_dummy': [273.15, 300.0, 7860.0, 0.9, 0.9, 5.82e8, 4.65e8, 0.325, 0.008, 1.3, 1790.0, 1.0],
        'inconel_718': [273.15, 300.0, 8242.0, 0.9, 0.9, 9.28e8, 9.79e8, 0.245847, 0.0056, 1.80073, 1623.15, 0.001],
    }

    total_depth = 1.0
    total_length = 1.0

    # Initialize logger, with log path and experiment name.
    exp_name = '4-18_active_inactive_pop_20_gens_250'
    log_path = '../../flexbench-data/4_exploration_pop_sizes/'
    logger = Logger(exp_name=exp_name, log_path=log_path)

    # Initialize task-performing system.
    tps = NSGAIIEstimator(
        logger=logger,
        n_process_params=3,
        range_process_params=[
            [0.1, 5.0],  # Range for cutting speed.
            [-0.5, 1.0],  # Range for cutting angle.
            [1.0e-6, 1.0e-3],  # Range for cutting depth.
        ],
        range_objectives=[
            [200.0, 10e6],  # Range for production time. (For ranges, see demo/oxley_simulation_range_objectives.py)
            [110.0, 7.72e223],  # Range for tool wear.
            [0.0, 500.0],  # Range for Fc.
            [0.0, 500.0],  # Range for Ft.
        ],
        pop_size=20,
        n_gens=250,
        use_tourn=True,
        tourn_size=2,
        eta_cross=30,
        eta_mut=20,
        load_percentage=1.0,
        target_percentage=1.0,
        epoch_length=5,
        gene_length=2,
        n_parallel=16,
        verbosity=False,
    )

    # Tasks for training and adaption, to be passed to task context.
    # Many tasks are included, comment to define one experiment.
    tasks_training = [
        # ('steel',),
        # ('tungsten_alloy',),
        # ('steel_dummy',),
        # ('inconel_718',),
        ('steel', 'tungsten_alloy'),
        ('steel', 'steel_dummy'),
        ('steel', 'inconel_718'),
        ('tungsten_alloy', 'steel_dummy'),
        ('tungsten_alloy', 'inconel_718'),
        ('steel_dummy', 'inconel_718'),
    ]

    tasks_adaption = [
        # (('steel',), 'tungsten_alloy'),
        # (('steel',), 'steel_dummy'),
        # (('steel',), 'inconel_718'),
        # (('tungsten_alloy',), 'steel'),
        # (('tungsten_alloy',), 'steel_dummy'),
        # (('tungsten_alloy',), 'inconel_718'),
        # (('steel_dummy',), 'steel'),
        # (('steel_dummy',), 'tungsten_alloy'),
        # (('steel_dummy',), 'inconel_718'),
        # (('inconel_718',), 'steel'),
        # (('inconel_718',), 'tungsten_alloy'),
        # (('inconel_718',), 'steel_dummy'),
        (('steel', 'tungsten_alloy'), 'steel_dummy'),
        (('steel', 'tungsten_alloy'), 'inconel_718'),
        (('steel', 'steel_dummy'), 'tungsten_alloy'),
        (('steel', 'steel_dummy'), 'inconel_718'),
        (('steel', 'inconel_718'), 'steel_dummy'),
        (('steel', 'inconel_718'), 'tungsten_alloy'),
        (('tungsten_alloy', 'steel_dummy'), 'steel'),
        (('tungsten_alloy', 'steel_dummy'), 'inconel_718'),
        (('tungsten_alloy', 'inconel_718'), 'steel'),
        (('tungsten_alloy', 'inconel_718'), 'steel_dummy'),
        (('steel_dummy', 'inconel_718'), 'steel'),
        (('steel_dummy', 'inconel_718'), 'tungsten_alloy'),
    ]

    # Initilize task context.
    task_context = TaskContextAdaption(
        logger=logger,
        params_names=params_names,
        tasks_params=tasks_params,
        total_depth=total_depth,
        total_length=total_length,
        tasks_training=tasks_training,
        tasks_adaption=tasks_adaption,
        # path_source = '../../flexbench-data/3_NSGA-II_varying_goals/3-2_varying_goals_active_inactive/logs/',
        path_target='../../flexbench-data/4_exploration_pop_sizes/4-16_baseline_pop_20_gens_250/logs/',
    )
    # task_context = TaskContextStudyParams(
    # logger=logger,
    # params_names=params_names,
    # tasks_params=tasks_params,
    # total_depth=total_depth,
    # total_length=total_length,
    # tasks_training=tasks_training,
    # tasks_adaption=tasks_adaption,
    # path_source = '',
    # path_target='../../flexbench-data/2_baseline_NSGA-II/2-2_tournament/logs/',
    # varied_params_names = ["eta_cross", "eta_mut"],
    # varied_params_names = ["pop_size"],
    # varied_params_names=['epoch_length', 'gene_length'],
    # varied_params_values={
    # "eta_cross": [20, 40, 80, 120, 140, 180],
    # "eta_mut": [20, 40, 80, 120, 140, 180],
    # "pop_size": [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # 'epoch_length': [5, 10, 15, 20, 25],
    # 'gene_length': [2, 3, 5],
    # },
    # include_target=False,
    # )

    # After initializing the logger, the TPS, and the task context, one should call task_context.run().
    # However, can't use multiprocess from inside "run" because using more than 16 subprocessess do not work in cluster.
    # Thus, a workaround is redefining tasks to be run for launching different parallel processes in slurm,
    # each for a task or group of tasks (first training tasks if required in adaption phase).
    # Need to call logger.save() below to create file "exp_info.txt" with the complete exp info.
    # Subsequent calls to logger.save() from inside run() do not overwrite it.
    # Also need to wait until process is running for changing task and launching next process,
    # otherwise both processess will use last tasks definition.

    logger.save(tps.save_solutions)
    n_runs = 100
    task_context.run(tps, n_runs=n_runs)

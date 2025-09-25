"""
Boolean circuit benchmark used in

Merav Parter, Nadav Kashtan, Uri Alon (2008):
Faciliated Variation: How Evolution Learns from Past Environments
To Generalize to New Environments.
PLOS Computational Biology 4(11)
https://doi.org/10.1371/journal.pcbi.1000206
"""
import typing
from inspect import signature
import numpy as np
from abc import ABC, abstractmethod

from flexbench.core import OracleTask


def EQ(x_1: bool, x_2: bool) -> bool:
    return x_1 == x_2


def AND(x_1: bool, x_2: bool) -> bool:
    return x_1 and x_2


def XOR(x_1: bool, x_2: bool) -> bool:
    return x_1 != x_2


def OR(x_1: bool, x_2: bool) -> bool:
    return x_1 or x_2


def target_generator(outer, inner1, inner2):
    return lambda x_1, x_2, x_3, x_4: outer(inner1(x_1, x_2), inner2(x_3, x_4))


class BooleanFunctionReconstructionTask(OracleTask):
    """Boolean function reconstruction task where training data equals test data."""

    def __init__(self, target):
        self._target = target
        self._target_arity = len(signature(target).parameters)

        self._input_space = self._create_input_space()

        self._X = self.input_space  # sklearn style
        self._y = self._evaluate_target()

    def _create_input_space(self):
        input_space = []
        for i in range(1 << self._target_arity):
            input_vector = []
            for j in range(self._target_arity - 1, -1, -1):
                input_vector.append(bool(i >> j & 1))
            input_space.append(input_vector)
        return input_space

    def _evaluate_target(self):
        y = []
        for x in self.input_space:
            y.append(self._target(*x))
        return y

    @property
    def params(self):
        return {'oracle': self.oracle}

    @property
    def oracle(self):
        return self._target

    @property
    def performance_measure(self):
        return lambda y_pred, y: sum([x1 == x2 for x1, x2 in zip(y_pred, y)]) / len(y_pred)

    @property
    def success_criterion(self):
        return lambda performance: performance >= 1.0

    @property
    def input_space(self):
        return self._input_space

    def run(self, tps, adapt=False):
        tps.performance_measure = self.performance_measure
        tps.success_criterion = self.success_criterion
        tps.fit(self._X, self._y, adapt=adapt)
        y_pred = tps.predict(self._X)
        print('Performance: ', self.performance_measure(y_pred, self._y))


class TaskContextFixedGoal:
    def __init__(self, logger):
        targets_training = [
            target_generator(AND, EQ, EQ),
            target_generator(AND, EQ, XOR),
            target_generator(OR, XOR, XOR),
        ]

        targets_adaption = [target_generator(AND, EQ, EQ), target_generator(AND, EQ, EQ), target_generator(AND, EQ, EQ)]

        self._tasks_training = [BooleanFunctionReconstructionTask(target) for target in targets_training]
        self._tasks_adaption = [BooleanFunctionReconstructionTask(target) for target in targets_adaption]

        self._tasks_names = ['AND_EQ_EQ_to_AND_EQ_EQ', 'AND_EQ_XOR_to_AND_EQ_EQ', 'OR_XOR_XOR_to_AND_EQ_EQ']

        # Register task context into logger.
        task_context_desc = (
            'Training under fixed goal on a source boolean circuit and adaption to a target boolean circuit.\nCombinations: \
             \n%s\n%s\n%s\nRepetitions=2'
            % (self._tasks_names[0], self._tasks_names[1], self._tasks_names[2])
        )

        self.logger = logger
        self.logger.register_task_context(task_context_desc)

    def run(self, tps):
        # Run one repetition for each training/adaption, register results into logger.
        for idx in range(len(self._tasks_training)):
            task_training = self._tasks_training[idx]
            task_adaption = self._tasks_adaption[idx]
            task_name = self._tasks_names[idx]

            for _ in range(2):
                task_training.run(tps, adapt=False)
                tps.register_data(task_name, 'training')

                task_adaption.run(tps, adapt=True)
                tps.register_data(task_name, 'adaption')

        # Save log information.
        self.logger.save(tps.save_solutions)


class BooleanFunctionEstimator(ABC):
    """An abstract TPS (Task Performing System) class defining the minimal interface of a task-performing system required."""

    @abstractmethod
    def __init__(self, logger, n_inputs: int = 4, n_nand_gates: int = 12):
        """Create a new instance.

        Args:
            logger: Logger
                Logger object for all experimental information and data.
            n_inputs: int, default: 4
                The number of input gates of the logic circuit (i.e., the arity of the Boolean function to learn)
            n_nand_gates: int, default: 12
                The number of NAND gates the logic circuit maximally consists of.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, y, adapt=False):
        """Learn a Boolean function based on the training samples.

        Args:
            X: array-like[bool] of shape (n_samples, n_inputs)
                The list of training Boolean input vectors.
            y: list[bool] of length n_samples
                The list of training Boolean out values.
            adapt: Union(bool, str)
                False if running from scratch, True if using solution from previous run, str if loading solution from somewhere.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Evaluate learned logic circuit.

        Args:
            X: array-like[bool] of shape (n_samples, n_inputs)
                A list of Boolean input vectors.

        Returns:
            y: list[bool]
                The computed Boolean outputs of the learned logic circuit.
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

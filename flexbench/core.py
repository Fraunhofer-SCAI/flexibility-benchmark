"""Core API"""
import typing, os, csv, pickle
from abc import ABC, abstractmethod

from flexbench.exceptions import NoCostError, NoSolutionError


class Task(ABC):
    """Base class for all tasks."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def performance_measure(self):
        """Returns the performance measure of the task."""
        raise NotImplementedError

    @property
    @abstractmethod
    def success_criterion(self):
        'Returns the function that computes if a task-performing system has successfully performed a task.'
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError

    def get_params(self):
        return self.params

    @abstractmethod
    def run(self, task_performing_system, adapt=False):
        """Let a task-performing system do the task and measure its performance.

        Args:
            task_performing_system
            adapt: False for running from scratch, True for using solution stored in task performing system, str for loading a solution.
                   Task performing system should process this.

        Returns:
            run: an object representing the run
        """
        raise NotImplementedError


class OracleTask(Task, ABC):
    """Base class for all tasks where an oracle is available."""

    @property
    @abstractmethod
    def oracle(self):
        raise NotImplementedError


class SupervisedLearningTask(Task, ABC):
    """Base class for all supervised learning tasks."""

    @property
    @abstractmethod
    def training_data(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def test_data(self):
        raise NotImplementedError


class ReinforcementLearningTask(Task, ABC):
    """Base class for all reinforcement learning tasks."""

    @property
    @abstractmethod
    def environment(self):
        raise NotImplementedError


class OptimizationTask(Task, ABC):
    """Base class for all optimization tasks."""

    @property
    @abstractmethod
    def objective_function(self):
        """Returns the objective function of the optimization task."""
        raise NotImplementedError


class Logger:
    """Class that holds information from an experiment.

    Parameters
    ----------
    exp_name: str
        Identifier of the experiment.
    log_path: str
        Where to save all information after experiment is done, or in between.

    Attributes
    ----------
    exp_info: dict
        Dictionary containing all data that describes the experiment (exp_name, log_path, system_desc, task_context_desc, etc).
    data: dict
        Dictionary containing all data from the experiment (training and adaption).
    """

    def __init__(self, exp_name, log_path):
        try:
            os.mkdir(log_path + exp_name)
        except FileExistsError:
            pass
        try:
            os.mkdir(log_path + exp_name + '/logs/')
        except FileExistsError:
            pass

        self.exp_name = exp_name
        self.log_path = log_path + exp_name + '/logs/'

        self.exp_info = {}
        self.exp_info['exp_name'] = self.exp_name
        self.exp_info['log_path'] = self.log_path

        self.data = {}

    def register_system(self, system_desc):
        """Registers a system description.

        Args:
            system_desc: str
                Description of the system, that should contain also the parameters used for reproducibility.
        """
        self.exp_info['system_desc'] = system_desc

    def register_task_context(self, task_context_desc):
        """Registers a task context description.

        Args:
            task_context_desc: str
                Description of the task context, with experiment logic for training and adaption phases for reproducibility.
        """
        self.exp_info['task_context_desc'] = task_context_desc

    def reset_data(self):
        """Resets the data stored, but keeps the information identifying the experiment.
        Used for saving data in between runs if one wants to save memory.
        """
        self.data = {}

    def register_data(self, task, exp_phase, data_id, data):
        """Registers some specific data to the training or adaption dictionary.

        Args:
            task: str
                Identifier of the current task.
            exp_phase: str
                "training" for training phase and "adaption" for adaption phase.
            data_id: str
                Data identifier.
            data: Sequence
                Data to be stored.
        """

        if not task in self.data:
            self.data[task] = {}
            self.data[task]['training'] = {}
            self.data[task]['adaption'] = {}

        if not data_id in self.data[task][exp_phase]:
            self.data[task][exp_phase][data_id] = []

        # Adds to a list to support multiple repetitions on a single task.
        self.data[task][exp_phase][data_id].append(data)

    def save(self, save_solutions):
        """Saves all information from experiment.

        Args:
            save_solutions: func
                Function provided to save solutions stored.
        """

        # Save experiment information.
        if 'exp_info.txt' not in os.listdir(self.log_path):
            with open(self.log_path + 'exp_info.txt', 'w') as f:
                f.write('Experiment name:\n%s\n\n' % self.exp_info['exp_name'])
                f.write('Log path:\n%s\n\n' % self.exp_info['log_path'])
                f.write('Task-performing system:\n%s\n\n' % self.exp_info['system_desc'])
                f.write('Task context:\n%s' % self.exp_info['task_context_desc'])

        # Save training and adaption logs.
        for task in self.data.keys():
            try:
                os.mkdir(self.log_path + task + '/')
            except FileExistsError:
                pass

            for log_type in ['training', 'adaption']:
                if self.data[task][log_type]:
                    task_path = self.log_path + task + '/' + log_type + '/'

                    try:
                        os.mkdir(task_path)
                    except FileExistsError:
                        pass

                    self.save_task(task_path, self.data[task][log_type], save_solutions)

        print('Logs saved in %s' % self.log_path)

    def save_task(self, task_path, task_logs, save_solutions):
        """Saves information from a specific task.

        Args:
            task_path: str
                Path where to save data.
            task_logs: dict
                Dictionary with log data. Each key is a log file, each value is a list for multiple repetitions. As a standard, each repetition is saved 			as a row in a .csv file, and the "solution" file is saved via a provided function. For specific needs, the method can be reimplemented
                in a Logger class that inherits from this class.
            save_solutions: func
                Function provided for saving the solutions that were stored.
        """

        # Cost is an obligatory field.
        if task_logs['cost']:
            with open(task_path + 'cost.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(task_logs['cost'])
        else:
            raise NoCostError('There is no cost information saved in logger.')

        # Solution is an obligatory field.
        if task_logs['solution']:
            # Create dedicated solutions directory.
            try:
                os.mkdir(task_path + 'solutions/')
            except FileExistsError:
                pass

            for solutions in task_logs['solution']:
                # Read how many files there are (one for each repetition) and set number to current file.
                nRep = (
                    len(
                        [
                            name
                            for name in os.listdir(task_path + 'solutions/')
                            if name.startswith('solutions_repetition_')
                        ]
                    )
                    + 1
                )
                save_solutions(solutions, task_path + 'solutions/solutions_repetition_' + str(nRep))

        else:
            raise NoSolutionError('There is no solution information saved in logger.')

        # Extra files.
        for k in task_logs.keys():
            if k != 'cost' and k != 'solution':
                with open(task_path + k + '.csv', 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(task_logs[k])

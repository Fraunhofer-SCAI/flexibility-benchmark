# FlexProbBench

A testbench framework for flexible task-performing systems. Defines an API for flexibility experiments, to which one can add different flexibility benchmarks composed of related tasks and/or task-performing systems to be evaluated in a structured way.

## Installation

After cloning the repository, pip install the package in editable state once with:

```bash
pip install -e .
```

This repository uses [pre-commit hooks](https://pre-commit.com/). Installation and enabling of pre-commit in a cloned repository is taken care of when installing the package (in `setup.py`).

At the moment, code is configured to store experimental and analysis results in a directory named `flexbench-data` placed on the same level in the directory to where `flexbench` is cloned. 

## Repository Structure

This repository is structured as follows.

### flexbench

Where the API and benchmarks are defined. In the context of a flexibility experiment, this API provides an interface for the concepts of **Task**, **Task Context**, and **Task Performing System**. Furthermore, it provides a structured way to conduct such experiments and to log results. `core.py` provides the interface for an abstract task class and provides a **Logger** object, that helps structuring the experiment. In `benchmarks`, one should put concrete benchmarks, which consist of a concrete implementation of a task, one or more task contexts, where the experimental logic is defined, and an abstract class defining the interface of an estimator for solving that given task.

### experiments

Where the actual flexibility experiments go. More on how all comes together [here](#flexibility-experiment).

### results

Were the experimental logs detailing the experiments that were performed are stored, together with the Jupyter notebooks for analysis of results.

### demo

Contains demonstrations of functionalities of the API, like how to run a basic flexibility experiment or how to use some functionality or calculate something related to a given benchmark.

## Flexibility Experiment

The main goal of this repository and the API it defines is to run structured flexibility experiments with different benchmarks and task-performing systems. A _flexibility experiment_ as such is composed by the following elements:

* Task;
* Task Context;
* Task-performing System.

A concrete task and task context should be defined in `flexbench/benchmarks/`, following the interface for the task provided in `flexbench/core.py`. In the same file, an interface for a task-performing system that solves the task is defined. Finally, one should also define one or more task contexts, that apply the task-performing system to different tasks and performs the logic of the flexibility experiment.

In an experiment, defined in `experiments/`, one provides a concrete implementation for a task-performing system or imports it from some external library and add the interface defined. Then, we need to import the task context and a logger (defined in `flexbench/core.py`). This logger is registered both in the task-performing system and in the task context and structures the log data that is stored after or while the experiment runs. A simple demonstration of how all comes together is shown in `demo/logic_circuit_demo.py`, that uses a genetic algorithm on the benchmark defined in `flexbench/benchmarks/logic_circuit.py`.

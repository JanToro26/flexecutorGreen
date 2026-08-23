import logging
import operator
import os
import pickle
import random
from typing import Dict

import numpy as np
from deap import algorithms, base, creator, gp, tools
from overrides import overrides
from scipy.optimize import differential_evolution

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, ConfigBounds, StageConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flexecutor.workflow.stage import Stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def rand101():
    return random.randint(-1, 1)


class GAPerfModel(PerfModel):
    def __init__(
        self,
        stage: Stage,
        population_size=300,
        crossover_prob=0.7,
        mutation_prob=0.2,
        n_generations=40,
    ):
        super().__init__("genetic", stage)
        self._population_size = population_size
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._n_generations = n_generations
        self._data = None
        self._best_individual = None
        self._toolbox = base.Toolbox()
        self._setup_genetic_algorithm()

    def _setup_genetic_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        pset = gp.PrimitiveSet("MAIN", 3)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addEphemeralConstant("rand101", rand101)
        pset.renameArguments(ARG0="cpus", ARG1="memory", ARG2="workers")

        self._toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self._toolbox.register(
            "individual", tools.initIterate, creator.Individual, self._toolbox.expr
        )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        self._toolbox.register("compile", gp.compile, pset=pset)

        self._toolbox.register("evaluate", self._evaluate)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register(
            "mutate", gp.mutUniform, expr=self._toolbox.expr, pset=pset
        )
        self._toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self._toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

    @overrides
    def save_model(self):
        folder = "/".join(self._model_dst.split("/")[:-1])
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self._model_dst, "wb") as file:
            pickle.dump(self._best_individual, file)

    @overrides
    def load_model(self):
        with open(self._model_dst, "rb") as file:
            self._best_individual = pickle.load(file)

    def optimize(self, bounds: ConfigBounds) -> StageConfig:
        objective_func = self._objective_func

        def integer_objective_func(x):
            x_int = np.round(x).astype(int)
            return objective_func(x_int)

        res = differential_evolution(
            integer_objective_func,
            bounds.to_tuple_list(),
            strategy="best1bin",
            mutation=(0.5, 1),
            recombination=0.7,
            disp=True,
        )
        return StageConfig(*np.round(res.x).astype(int))

    def _evaluate(self, individual):
        func = self._toolbox.compile(expr=individual)
        errors = []
        for cpus, memory, workers, actual in self._data:
            try:
                predicted = func(cpus, memory, workers)
                if not np.isfinite(predicted) or predicted > 100 or predicted <= 0:
                    penalty = 1e10
                else:
                    penalty = (predicted - actual) ** 2
                errors.append(penalty)
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                errors.append(1e10)
        return (np.mean(errors),)

    def train(self, stage_profile_data: Dict) -> None:
        def preprocess_profiling_data(profiling_data):
            processed_data = []
            cold_starts = []

            for config, data in profiling_data.items():
                num_vcpu, memory, num_func = config

                config_key = (num_vcpu, memory, num_func)

                cold_starts.extend([np.mean(times) for times in data["cold_start"]])

                read_times = [np.mean(times) for times in data["read"]]
                compute_times = [np.mean(times) for times in data["compute"]]
                write_times = [np.mean(times) for times in data["write"]]

                latencies = [
                    sum(times) + np.mean(cold_starts)
                    for times in zip(read_times, compute_times, write_times)
                ]

                for latency in latencies:
                    processed_data.append(
                        (config_key[0], config_key[1], config_key[2], latency)
                    )

            return processed_data

        self._data = preprocess_profiling_data(stage_profile_data)
        print(self._data)
        pop = self._toolbox.population(n=self._population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(
            pop,
            self._toolbox,
            cxpb=self._crossover_prob,
            mutpb=self._mutation_prob,
            ngen=self._n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
        self._best_individual = hof[0]

        def objective_func(x):
            cpus, memory, workers = np.round(x).astype(int)
            try:
                value = self._toolbox.compile(expr=self._best_individual)(
                    cpus, memory, workers
                )
                if not np.isfinite(value):
                    raise ValueError("Non-finite value")
                if value > 100 or value <= 0:
                    return 1e10
                return value
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1e10

        self._objective_func = objective_func

    @property
    @overrides
    def parameters(self):
        return "Yet to be implemented"

    def predict_time(self, config) -> FunctionTimes:
        func = self._toolbox.compile(expr=self._best_individual)
        try:
            return FunctionTimes(total=func(config.cpu, config.memory, config.workers))
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            return FunctionTimes()

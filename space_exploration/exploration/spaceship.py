import random
import time

from space_exploration.exploration.logging import log
from space_exploration.exploration.precision_level import PrecisionLevel


class SpaceShip:
    def __init__(self, model, point, precision):
        self.precision = precision
        self.point = point
        self.model = model
        self.cumulative_search_time = 0
        self.iteration = 0

    def probe(self):
        start = time.time()
        time.sleep(self.simulate_duration())
        result = self.simulate_result()
        self.cumulative_search_time += time.time() - start
        log(result, self.point, self.precision, self.cumulative_search_time, self.iteration)
        self.iteration += 1

    def simulate_duration(self) -> float:
        return {PrecisionLevel.FASTEST: 0.1,
                PrecisionLevel.FAST: 0.5,
                PrecisionLevel.NORMAL: 1.0}[self.precision]

    def simulate_result(self) -> float:
        noise = {PrecisionLevel.FASTEST: 5.0,
                 PrecisionLevel.FAST: 2.0,
                 PrecisionLevel.NORMAL: 0.5}[self.precision]
        true_score = sum([v**2 for v in self.point.values()])
        return true_score + random.uniform(-noise, noise)
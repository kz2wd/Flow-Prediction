from space_exploration.exploration.precision_level import PrecisionLevel

exploration_log = []

def log(score: float, point: dict, precision: PrecisionLevel, duration: float, iteration: int):
    exploration_log.append({
        'point': point,
        'score': score,
        'precision': precision,
        'time': duration,
        'iteration': iteration,
    })
"""Generation loop coordinating SAGE gates and constrained NSGA-II."""

import random

from mogpu.search.archive import ParetoArchive
from mogpu.search.nsga2 import environmental_selection
from mogpu.search.records import CandidateRecord
from mogpu.search.sage_pareto import f0


class EvolutionController:
    def __init__(self, evaluator, protocol: dict):
        self.evaluator = evaluator
        self.protocol = protocol
        self.archive = ParetoArchive(protocol["archive_capacity"])
        self.rng = random.Random(protocol["random_seed"])

    def run_generation(
        self, population: list[CandidateRecord]
    ) -> list[CandidateRecord]:
        evaluated = []
        for record in population:
            record = f0(record)
            if record.status == "passed":
                record = self.evaluator(record)
            evaluated.append(record)
            self.archive.add(record)
        return environmental_selection(evaluated, self.protocol["population_size"])

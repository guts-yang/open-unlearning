"""Generation loop coordinating SAGE gates and constrained NSGA-II."""

from __future__ import annotations

import random

from mogpu.search.archive import ParetoArchive
from mogpu.search.nsga2 import environmental_selection
from mogpu.search.offspring import expand_initial, make_offspring
from mogpu.search.records import CandidateRecord
from mogpu.search.sage_pareto import action_evidence, f0, spec_from_record
from mogpu.search.scheduler import SearchScheduler
from trainer.unlearn.mogpu_dsl.gates import probe_action_evidence


class EvolutionController:
    def __init__(
        self, evaluator, protocol: dict, scheduler: SearchScheduler | None = None
    ):
        self.evaluator = evaluator
        self.protocol = protocol
        self.archive = ParetoArchive(protocol["archive_capacity"])
        self.rng = random.Random(protocol["random_seed"])
        self.scheduler = scheduler or SearchScheduler(self._f2)

    def run(self, initial: list[CandidateRecord]) -> list[CandidateRecord]:
        parents = expand_initial(initial, self.protocol, self.rng)
        evaluated = [self._sage_evaluate(record) for record in parents]
        for record in evaluated:
            self.archive.add(record)
        parents = environmental_selection(evaluated, self.protocol["population_size"])
        for generation in range(1, self.protocol["max_generations"]):
            children = make_offspring(parents, generation, self.protocol, self.rng)
            scored = [self._sage_evaluate(record) for record in children]
            for record in scored:
                self.archive.add(record)
            parents = environmental_selection(
                parents + scored, self.protocol["population_size"]
            )
        return parents

    def run_generation(
        self, population: list[CandidateRecord]
    ) -> list[CandidateRecord]:
        evaluated = [self._sage_evaluate(record) for record in population]
        for record in evaluated:
            self.archive.add(record)
        return environmental_selection(evaluated, self.protocol["population_size"])

    def _sage_evaluate(self, record: CandidateRecord) -> CandidateRecord:
        record = f0(record)
        if record.status != "passed":
            return record
        record = action_evidence(
            record, probe_action_evidence(spec_from_record(record))
        )
        if record.status != "passed":
            return record
        f2 = self.protocol["sage"]["f2"]
        max_steps = int(f2["budget"]["max_steps"])
        seed = int(self.protocol.get("training_seed", 0))
        key = (record.candidate_hash, max_steps, seed, "search")
        scored, _ = self.scheduler.evaluate(key, record)
        return scored

    def _f2(self, record: CandidateRecord) -> CandidateRecord:
        f2 = self.protocol["sage"]["f2"]
        self.evaluator.recipe["stage_budget"] = dict(f2["budget"])
        self.evaluator.recipe["checkpoint_steps"] = list(
            f2.get("checkpoint_steps", [f2["budget"]["max_steps"]])
        )
        self.evaluator.recipe["training_seed"] = self.protocol.get("training_seed", 0)
        return self.evaluator(record)

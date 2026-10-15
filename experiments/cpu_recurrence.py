"""Offline, bounded comparison of one versus two shared-block passes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from core import (
    Config, ModelRepository, log_run, print_metadata, source_provenance,
)
import train


DOCUMENTS = {
    "train": [
        f"A {color} robot found a {shape}. The robot put the {shape} on a shelf."
        for color in ("red", "blue", "green", "yellow")
        for shape in ("cube", "ball", "tile", "ring")
    ],
    "validation": [
        f"A {color} robot found a {shape}. The robot put the {shape} on a shelf."
        for color in ("orange", "purple")
        for shape in ("cube", "ball", "tile", "ring")
    ],
}


def document_hash(documents):
    payload = json.dumps(documents, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_synthetic(config, split, metadata):
    documents = DOCUMENTS[split]
    metadata[split] = {"version": "1", "fingerprint": document_hash(documents)}
    return documents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="artifacts/cpu-recurrence")
    args = parser.parse_args(argv)
    if torch.cuda.is_available():
        parser.error("This comparison requires CPU; set CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    output = Path(args.output_directory)
    provenance = source_provenance()
    results = []
    for passes in (1, 2):
        config = Config(
            dim=32, target_head_dim=8, context=32, batch_size=4,
            transform_layers=2, recurrent_steps=passes, num_experts=4, top_k=2,
            target_training_bytes=None, steps=200, validate_every=50,
            validation_batches=5, final_validation_batches=20,
            dataset_name="tinylm-synthetic-robots-v1",
            dataset_revision=document_hash(DOCUMENTS),
            train_split="train", validation_split="validation",
            model_directory=str(output / "model"), log_directory=str(output / "logs"),
        )
        repository = ModelRepository(config.model_directory, f"r{passes}-")
        repository.check_overwrite()
        with log_run(
            config.log_directory, f"r{passes}-", "CPU recurrence experiment",
            provenance=provenance,
        ) as log_path:
            started = time.perf_counter()
            train.train(
                config, repository, started,
                provenance=provenance, source_loader=load_synthetic,
            )
            runtime = time.perf_counter() - started
            model, best = repository.load(repository.best_filename, "cpu")
            _, last = repository.load(repository.last_filename, "cpu")
            result = {
                "recurrent_passes": passes,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "training_invocation_seconds": runtime,
                "best_step": best["step"],
                "best_metrics": best["validation_metrics"],
                "final_metrics": last["validation_metrics"],
                "cuda_peak_memory_mib": None,
                "cpu_peak_memory_mib": None,
                "log": str(log_path),
                "config": asdict(config),
                "provenance": best["provenance"],
            }
            print_metadata("Experiment result", result)
            results.append(result)
    summary = output / "results.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

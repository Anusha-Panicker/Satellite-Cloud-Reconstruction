
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np

root = Path(__file__).resolve().parents[1]

out = root / "outputs" / "large_patch_dataset"

source = out / "candidate_spectral_audit.csv"

targets = {
    "train": {
        "land_or_mixed": 380,
        "water_like": 120
    },
    "val": {
        "land_or_mixed": 57,
        "water_like": 18
    },
    "test": {
        "land_or_mixed": 57,
        "water_like": 18
    }
}

with open(source, newline="") as file:
    records = list(csv.DictReader(file))

if not records:
    raise ValueError("Spectral audit CSV is empty")

blocks = defaultdict(list)

for record in records:

    score = float(record["green_nir_contrast"])

    if score > 0.20:
        record["surface_proxy"] = "water_like"
    else:
        record["surface_proxy"] = "land_or_mixed"

    block = (
        int(record["block_x"]),
        int(record["block_y"])
    )

    blocks[block].append(record)

keys = sorted(blocks)

classes = ["land_or_mixed", "water_like"]

block_counts = {
    block: {
        cls: sum(
            row["surface_proxy"] == cls
            for row in members
        )
        for cls in classes
    }
    for block, members in blocks.items()
}

rng = np.random.default_rng(2026)

assignment = None

for attempt in range(20000):

    order = rng.permutation(len(keys))

    cursor = 0
    proposal = {}

    for split in ["val", "test"]:

        proposal[split] = []

        tally = {
            "land_or_mixed": 0,
            "water_like": 0
        }

        while any(
            tally[cls] < targets[split][cls]
            for cls in classes
        ):

            if cursor >= len(order):
                break

            block = keys[int(order[cursor])]
            cursor += 1

            proposal[split].append(block)

            for cls in classes:
                tally[cls] += block_counts[block][cls]

    proposal["train"] = [
        keys[int(i)]
        for i in order[cursor:]
    ]

    sufficient = all(
        sum(
            block_counts[block][cls]
            for block in proposal[split]
        ) >= targets[split][cls]
        for split in targets
        for cls in classes
    )

    if sufficient:
        assignment = proposal
        break

if assignment is None:
    raise RuntimeError(
        "Could not find a balanced geographic split"
    )

selected = []

for split in ["train", "val", "test"]:

    subset = [
        record
        for block in assignment[split]
        for record in blocks[block]
    ]

    for cls, quota in targets[split].items():

        pool = [
            record
            for record in subset
            if record["surface_proxy"] == cls
        ]

        rng.shuffle(pool)

        for record in pool[:quota]:

            result = record.copy()

            result["original_split"] = result["split"]
            result["split"] = split

            selected.append(result)

selected.sort(
    key=lambda record: record["patch_id"]
)

expected_total = sum(
    sum(quotas.values())
    for quotas in targets.values()
)

assert len(selected) == expected_total

assert len({
    record["patch_id"]
    for record in selected
}) == len(selected)

for first in targets:
    for second in targets:
        if first != second:
            assert not (
                set(assignment[first])
                & set(assignment[second])
            )

csv_path = out / "balanced_candidate_catalog.csv"

fieldnames = list(records[0].keys()) + [
    "original_split"
]

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(selected)

print("\nBALANCED GEOGRAPHIC SPLITS")
print("=" * 45)

for split in ["train", "val", "test"]:

    subset = [
        row for row in selected
        if row["split"] == split
    ]

    water_count = sum(
        row["surface_proxy"] == "water_like"
        for row in subset
    )

    land_count = len(subset) - water_count

    print(f"\n{split.upper()}")
    print(f"Land/mixed candidates: {land_count}")
    print(f"Water-like candidates: {water_count}")
    print(f"Total: {len(subset)}")
    print(f"Geographic blocks: {len(assignment[split])}")

print("\nTotal selected:", len(selected))
print("Remaining candidates:", len(records) - len(selected))
print("Saved:", csv_path)
print("\nGeographic split balancing completed.")

from pathlib import Path
from itertools import combinations
import csv
import math

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "outputs" / "large_patch_dataset" / "balanced_candidate_catalog.csv"
PATCH_SIZE = 256
BUFFER_PIXELS = (256, 512)

with CATALOG.open(newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

if not records:
    raise ValueError(f"Empty catalog: {CATALOG}")

expected_splits = ("train", "val", "test")
by_split = {s: [] for s in expected_splits}
for row in records:
    split = row["split"].strip()
    if split not in by_split:
        raise ValueError(f"Unknown split: {split}")
    by_split[split].append({
        "id": row["patch_id"],
        "col": int(row["col"]),
        "row": int(row["row"]),
        "block": (int(row["block_x"]), int(row["block_y"])),
    })


if len({r["id"] for rows in by_split.values() for r in rows}) != len(records):

    raise ValueError("Duplicate patch_id values found")

print("SPATIAL SPLIT SEPARATION AUDIT")
print("=" * 45)
print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE} pixels")
print("Distances are gaps between patch boundaries, not centers.")
print("A 256-pixel gap is about 1.28 km at the scene's 5 m pixel spacing.\n")

for a, b in combinations(expected_splits, 2):
    first, second = by_split[a], by_split[b]
    blocks_a = {r["block"] for r in first}
    blocks_b = {r["block"] for r in second}
    shared_blocks = blocks_a & blocks_b
    if shared_blocks:
        raise ValueError(f"Geographic blocks shared across {a}/{b}: {shared_blocks}")

    counts = {buffer: 0 for buffer in BUFFER_PIXELS}
    nearest = []

    for left in first:
        for right in second:
            horizontal_gap = max(0, abs(left["col"] - right["col"]) - PATCH_SIZE)
            vertical_gap = max(0, abs(left["row"] - right["row"]) - PATCH_SIZE)
            distance = math.hypot(horizontal_gap, vertical_gap)
            for buffer in BUFFER_PIXELS:
                if distance < buffer:
                    counts[buffer] += 1
            nearest.append((distance, left["id"], right["id"], left["block"], right["block"]))

    nearest.sort()
    print(f"{a.upper()} vs {b.upper()}")
    print(f"  Candidates: {len(first)} vs {len(second)}")
    print(f"  Shared geographic blocks: {len(shared_blocks)}")
    if nearest:
        print(f"  Minimum edge-to-edge gap: {nearest[0][0]:.1f} px")
        for buffer, count in counts.items():
            print(f"  Cross-split patch pairs closer than {buffer} px: {count}")
        print("  Five closest pairs:")
        for gap, id_a, id_b, block_a, block_b in nearest[:5]:
            print(f"    {id_a} {block_a} vs {id_b} {block_b}: {gap:.1f} px")
    print()

print("CHECK FINISHED")
print("Counts above are proximity warnings, not proof of duplicated pixels.")
print("Do not change existing split assignments until we've reviewed these results.")

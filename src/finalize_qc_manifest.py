from pathlib import Path
from collections import Counter
import csv

root = Path(__file__).resolve().parents[1]
meta = root / 'data' / 'processed' / 'cloud_reconstruction_v1' / 'metadata'
qc_path = meta / 'manual_qc_review.csv'
original_path = meta / 'patch_manifest.csv'


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])

qc, qc_columns = read_csv(qc_path)
manifest, manifest_columns = read_csv(original_path)
required_qc = {'patch_id', 'split', 'manual_status'}
required_manifest = {'patch_id', 'split'}
if not required_qc.issubset(qc_columns):
    raise ValueError(f'QC CSV missing {required_qc-set(qc_columns)}')
if not required_manifest.issubset(manifest_columns):
    raise ValueError(f'Manifest CSV missing {required_manifest-set(manifest_columns)}')
if len(qc) != 650 or len(manifest) != 650:
    raise ValueError(f'Expected 650 rows in each CSV; got QC={len(qc)}, manifest={len(manifest)}')

qc_by_id = {r['patch_id']:r for r in qc}
manifest_by_id = {r['patch_id']:r for r in manifest}
if len(qc_by_id) != 650 or len(manifest_by_id) != 650:
    raise ValueError('Duplicate patch IDs found')
if set(qc_by_id) != set(manifest_by_id):
    raise ValueError('Patch IDs do not match across QC and extraction manifest')

approved, excluded = [], []
extra_fields = ['manual_status','qc_priority','surface_proxy','issue','notes']
for r in manifest:
    patch_id=r['patch_id']
    decision=qc_by_id[patch_id]
    if decision['split'] != r['split']:
        raise ValueError(f'Split mismatch for {patch_id}')
    if decision['manual_status'] not in ('approved','uncertain','rejected'):
        raise ValueError(f'Pending/invalid QC decision: {patch_id}: {decision["manual_status"]}')
    enriched = r.copy()
    for key in extra_fields:
        if key in decision:
            enriched[key] = decision[key]
    if decision['manual_status']=='approved':
        approved.append(enriched)
    else:
        excluded.append(enriched)

fields=manifest_columns + [f for f in extra_fields if f not in manifest_columns]
for path,rows in [(meta/'approved_target_manifest.csv',approved), (meta/'excluded_target_manifest.csv',excluded)]:
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print('Saved:',path)

print('Approved:',len(approved),dict(Counter(r['split'] for r in approved)))
print('Excluded:',len(excluded),[(r['patch_id'],r['manual_status']) for r in excluded])
print('Note: approved means visually suitable for this prototype, NOT proven cloud-free.')

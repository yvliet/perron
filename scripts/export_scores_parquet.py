import sqlite3
import struct
from pathlib import Path
import pandas as pd

db_path = Path("results/gemma4_protocol/scores_cache.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens FROM pointwise_scores")
rows = cur.fetchall()
conn.close()

cleaned_rows = []
for instance_id, cand_id, prompt_variant, s, format_mass, n_prompt_tokens in rows:
    if isinstance(cand_id, bytes):
        if len(cand_id) == 8:
            cand_id = struct.unpack("<q", cand_id)[0]
        elif len(cand_id) == 4:
            cand_id = struct.unpack("<i", cand_id)[0]
        else:
            cand_id = int.from_bytes(cand_id, "little")
    else:
        cand_id = int(cand_id)
    cleaned_rows.append({
        "instance_id": instance_id,
        "candidate_id": cand_id,
        "prompt_variant": prompt_variant,
        "s": float(s),
        "format_mass": float(format_mass),
        "n_prompt_tokens": int(n_prompt_tokens),
    })

df = pd.DataFrame(cleaned_rows)
out_path = Path("data_release/scores.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)

print(f"Exported {len(df)} rows to {out_path} ({out_path.stat().st_size} bytes)")
print("Schema:")
print(df.dtypes)
print("Sample:")
print(df.head(5))

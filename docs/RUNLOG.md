# Perron Evaluation & Run Log

| UTC | command | split | git sha | output file | note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-10-03T06:51:27.847717+00:00 | python scripts/run_heldout.py --suite table2 --seed 20261003 | heldout | 856b88cdfb2337556a751ed0d6de1cafed89a932 | results/heldout_table2_v2_postaudit.json | post-audit heldout evaluation (Table 2) |
| 2026-10-03T07:02:56.757960+00:00 | python scripts/run_heldout.py --suite fusion --seed 20261003 | heldout | fa47fbe7a0fd51326b8fa170c6cdacc4a5bc7aab | results/heldout_fusion.json | post-audit heldout fusion evaluation |
| 2026-10-03T07:03:10.283696+00:00 | python scripts/run_heldout.py --suite strata --seed 20261003 | heldout | fa47fbe7a0fd51326b8fa170c6cdacc4a5bc7aab | results/heldout_strata.json | post-audit heldout strata evaluation |

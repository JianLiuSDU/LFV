# Legacy Backup

A lightweight copy of the pre-cleanup LFV repository was saved outside this
repository:

```text
/home/users1/ljian/LFV_legacy_20260727_no_data_no_third_party
```

The backup excludes:

- `.git`
- `data/`
- `third_party/`
- `__pycache__/`
- `*.pyc`

The current repository has intentionally removed old model/training/simulation
code so the new two-stage generation framework can be rebuilt cleanly.


# Paper NORM and fibre-mass experiments

Start with the [script-by-script guide](../../docs/SCRIPTS.md): it maps every
retained file to a paper result or a required helper, with inputs and outputs.
The [walkthrough](../../docs/CONDITIONING.md) gives copyable reproduction commands.

```bash
uv run --script --locked experiments/conditioning/suite.py list
```

The launcher lists runnable paper workflows with descriptions. Shared helpers
are staged automatically and are not offered as commands. Source hashes are
checked before running; work happens in a separate exported input directory.

- `scripts/`: Section 6, fibre appendix and their shared dependencies.
- `source_manifest.json`: historical source hashes and current packaged hashes.
- `suite.py.lock`: separate locked experiment environment.
- `environment.json`: observed original direct dependency versions.
- `../../artifacts/conditioning/`: reference tables, input manifests and upstream patch.

Large inputs and checkpoints remain external. Historical prepared archives and
shared validation routines retain extra arms to preserve the original frozen
selection audit. Standalone joint-only figures/searches and exploratory code
have been removed; see Git history for their earlier versions.

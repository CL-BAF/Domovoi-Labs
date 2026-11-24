# Contributing to Domovoi Labs

This repository owns implementation, configs, tests and executable tools. The [research repository](https://github.com/CL-BAF/Domovoi-research) owns experiment plans, evidence, interpretations and findings.

Before starting any research run, publish and approve its plan with the exact implementation commit. Before starting the next run, publish the prior run's result and interpretation, including failures and interruptions. Use the supported launchers and follow the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

Keep source changes reviewable and run the relevant correctness tests. Group routine development verification using the test suite and record the commands, environment and outcomes in the change review. Synthetic correctness fixtures may exercise training code. A measurement intended to support a research conclusion belongs in a protocol experiment.

Preserve failed artifacts and existing history. Keep credentials, private machine paths, corpora and large checkpoints out of Git. Publish the permitted evidence or its digest, provenance and recovery instructions in the research repository.

# Decoding Reversibility in a Finite Tool Sandbox

Srijan Chopra, Advanced Deep Neural Network.

Code and documentation for an individual measurement study using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. See CONTRIBUTIONS.md for AI assistance disclosure and SUBMISSION_STATUS.md for scientific limitations. Statements there about bundled evidence describe the separate submission archive, not this code-only repository.

## Repository scope

This repository includes Python scripts, notebooks with saved outputs removed, the data card, protocol documentation and observed dependency versions. Datasets, annotation responses, experiment logs, model weights, activations, reports and slides are excluded. They remain in the separate course submission. No claim of complete experimental coverage is made: ScienceWorld labelled data and cross-environment transfer, the combined 2,000-example target and the main-dataset stated-knowledge comparison remain incomplete.

## Running the code

Use Google Colab for the historical workflow. The scripts expect `/content/drive/MyDrive/ADNN_Irreversibility`. Mount your own Drive, install the required dependencies and review the input/output paths in each script before running. These scripts are not a single run-all pipeline. Several use timestamped input paths which must be reconciled with your own generated files.

Upload a Python script to Colab and execute it using `%run /content/SCRIPT_NAME.py`; this defines `__file__` correctly. Read any command-line options in that script first. GPU extraction needs a CUDA runtime. Llama requires approved model access. Store Hugging Face credentials in Colab Secrets as `HF_TOKEN`, never in source files.

## Workflow map

- `code/13_build_sandbox_candidate_colab.py`: generate the finite-tool dataset and reference recovery evidence.
- Scripts 25–27: preflight checks, activation extraction and main probes.
- Scripts 29–32: reduced controls, statistical summaries, ablations and audits. Script 28 is the historical extended-control procedure.
- Scripts 15 and 19–24: paired behaviour task setup, revised model runs, replay audits and stated-knowledge measurements. Read `docs/RQ3_PROTOCOL.md` and script dependencies.
- Scripts 34–36: AI annotation and agreement diagnostics. Completed human responses are separate inputs.
- Scripts 37–42: ScienceWorld feasibility and search coverage diagnostics. Unresolved searches do not establish irreversible labels.
- Earlier scripts/notebooks: development pilots and historical attempts, retained for provenance.

`requirements.txt` records observed versions, not a tested portable installation lock. Preserve a working CUDA-enabled PyTorch installation rather than replacing it blindly. ScienceWorld needs Java and the pinned revision listed in that file. Experiments need their generated predecessor outputs; this code-only repository does not contain saved results.

## 13 November continuation

Scripts 43 and 44 prepare the missing same-example RQ2 experiment and audit the pinned ScienceWorld state export. The RQ2 generation phase still needs a GPU. No new RQ2 model scores or ScienceWorld labels are claimed. See `docs/RESEARCH_COMPLETION_PLAN.md`. Run helper checks with `python3 code/test_rq2_helpers.py`. Prepared query packets and audit outputs remain in the separate local update, not this code-only repository.

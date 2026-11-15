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

## Follow-up experiments

- Script 43 runs and scores the sandbox same-example RQ2 comparison. Both model runs have been completed. The raw outputs and probe evidence are available in the course submission.
- Scripts 44–50 record the ScienceWorld feasibility checks. Scripts 51–52 generate and audit the reduced benchmark. Run them together in the code directory with the pinned ScienceWorld installation. See docs/SCIENCEWORLD_RESTRICTED_PROTOCOL.md for the exact scope.
- task3_qwen_colab.ipynb annotates the separate blinded JSON packet supplied with the course submission. The notebook requires a GPU and preserves its input hash with the responses.
- transfer_preflight.py audits class and group coverage before any transfer fitting. See docs/TRANSFER_FEASIBILITY.md and SUBMISSION_STATUS.md for the decision to stop at feasibility.

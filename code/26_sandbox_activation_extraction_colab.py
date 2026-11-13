"""GPU extraction for frozen sandbox candidate; no probe scores computed."""
import sys,json,hashlib,time,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from google.colab import drive,userdata
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
model_name=sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else 'qwen'
assert model_name in ('qwen','llama')
assert torch.cuda.is_available(),'Select a GPU runtime'
PROJECT=Path('/content/drive/MyDrive/ADNN_Irreversibility')
if not PROJECT.is_dir():drive.mount('/content/drive')
preflight=PROJECT/'data/sandbox_candidate_v2_20261018T095320799792Z/preflight_20261019T102812492342Z'
plan_raw=(preflight/'analysis_plan.json').read_bytes();plan_hash=hashlib.sha256(plan_raw).hexdigest()
plan=json.loads(plan_raw)
assert json.loads((preflight/'summary.json').read_text())['ready_for_extraction']
dataset_expected_hash='1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
assert plan['dataset_sha256']==dataset_expected_hash
MODEL_ID,REVISION=plan['models'][model_name]
expected_revisions={'qwen':'a09a35458c702b33eeacc393d103063234e8bc28','llama':'0e9e39f249a16976918f6564b8830bc894c89659'}
assert REVISION==expected_revisions[model_name]
SEED=42;set_seed(SEED)
versions={p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes','numpy')}
token=None
if model_name=='llama':
    token=userdata.get('HF_TOKEN').strip()
    assert token and not any(c.isspace() for c in token),'Invalid whitespace in HF_TOKEN'
print('Loading',MODEL_ID,flush=True)
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,token=token,use_fast=True)
assert tokenizer.is_fast
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,token=token,quantization_config=quant,dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
model.eval();del token

import hashlib, gc
SOURCE = PROJECT / "data/sandbox_candidate_v2_20261018T095320799792Z"
SPLITS = SOURCE / "private"
raw = (SOURCE / "sandbox_candidate.jsonl").read_bytes()
records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
assert len(records) == 1200 and len({r["id"] for r in records}) == 1200
assert dataset_expected_hash == hashlib.sha256(raw).hexdigest()
dataset_hash = hashlib.sha256(raw).hexdigest()
for split_seed in [42, 123, 2026]:
    manifest = json.loads((SPLITS / f"split_{split_seed}.json").read_text())
    assert manifest["dataset_sha256"] == dataset_hash
assert model.config._attn_implementation == "sdpa"
SYSTEM = "You are an agent acting in a simulated environment. Consider the task, current state, and available tools, then choose one tool call."
config = {
    "model": MODEL_ID, "revision": REVISION, "dataset_sha256": dataset_hash,
    "quantization": "NF4_double", "compute_dtype": "float16", "attention": "sdpa",
    "versions": versions, "system_prompt": SYSTEM, "prompt_version": "task_state_candidate_v1",
    "max_tokens": 2048, "token_summaries": ["last_action_token", "mean_action_tokens"],
    "storage_dtype": "float32", "seed": SEED,
    "scope": "sandbox_candidate_v2_only", "analysis_plan_sha256": plan_hash, "hidden_state_convention": "HF returned states: embedding at 0; final state after final normalization",
}
run_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
EXTRACTION = PROJECT / "activations" / (model_name + "_sandbox1200_" + run_key)
EXTRACTION.mkdir(parents=True, exist_ok=True)
config_path = EXTRACTION / "config.json"
if config_path.exists():
    assert json.loads(config_path.read_text()) == config
else:
    config_path.write_text(json.dumps(config, indent=2))
print("Records:", len(records))
print("Resume folder:", EXTRACTION)

assert SYSTEM == plan["system_prompt"]
(EXTRACTION / "analysis_plan.json").write_bytes(plan_raw)
(EXTRACTION / "extraction_source.py").write_text(Path(__file__).read_text())
completed = 0
resumed = 0
expected_shape = (model.config.num_hidden_layers + 1, model.config.hidden_size)
for index, record in enumerate(records, 1):
    record_id = record["id"]
    assert all(ch.isalnum() or ch in "_-" for ch in record_id)
    target = EXTRACTION / (record_id + ".npz")
    log_path = EXTRACTION / (record_id + ".json")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Task: " + record["task_description"] + "\n" + record["state_text"]},
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prefix + record["action_text"]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    reusable = False
    if target.exists() and log_path.exists():
        try:
            saved_log = json.loads(log_path.read_text())
            with np.load(target, allow_pickle=False) as saved:
                reusable = (saved_log["id"] == record_id and saved_log["prompt_sha256"] == prompt_hash
                            and saved_log["run_key"] == run_key
                            and all(saved[k].shape == expected_shape and np.isfinite(saved[k]).all() for k in ["last", "mean"]))
        except (ValueError, KeyError, OSError, EOFError):
            reusable = False
    if reusable:
        resumed += 1
    else:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        count = encoded["input_ids"].shape[1]
        assert count <= config["max_tokens"], f"{record_id}: {count} tokens; stop instead of truncating."
        positions = [i for i, (a,b) in enumerate(offsets) if b > len(prefix) and a < len(prompt)]
        assert positions and offsets[positions[-1]][1] == len(prompt)
        inputs = {k:v.to("cuda") for k,v in encoded.items()}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.base_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            states = output.hidden_states
            assert len(states) == expected_shape[0]
            bad = [i for i,h in enumerate(states) if not torch.isfinite(h).all().item()]
            assert not bad, f"{record_id}: nonfinite hidden states at {bad}"
            last = torch.stack([h[0,positions[-1],:].float().cpu() for h in states]).numpy()
            mean = torch.stack([h[0,positions,:].float().mean(dim=0).cpu() for h in states]).numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert last.shape == mean.shape == expected_shape
        assert np.isfinite(last).all() and np.isfinite(mean).all()
        log = {"id":record_id,"run_key":run_key,"prompt":prompt,"prompt_sha256":prompt_hash,
               "action_token_positions":positions,"sequence_tokens":int(count),
               "seconds":elapsed,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle: np.savez_compressed(handle, last=last, mean=mean)
        temporary.replace(target)
        log_temp = log_path.with_suffix(".json.tmp")
        log_temp.write_text(json.dumps(log, indent=2)); log_temp.replace(log_path)
        del output, states, inputs, encoded, last, mean
        completed += 1
    if index % 8 == 0 or index == len(records):
        print(f"{index}/{len(records)} ready ({completed} newly extracted, {resumed} resumed)", flush=True)
print("Extraction loop complete. Run the final verification cell.")

logs = []
for record in records:
    record_id = record["id"]
    with np.load(EXTRACTION / (record_id + ".npz"), allow_pickle=False) as saved:
        for name in ["last", "mean"]:
            assert saved[name].shape == expected_shape
            assert np.isfinite(saved[name]).all()
    log = json.loads((EXTRACTION / (record_id + ".json")).read_text())
    assert log["id"] == record_id and log["run_key"] == run_key
    logs.append(log)
summary = {
    "records_verified":len(logs),"shape_per_summary":list(expected_shape),
    "summaries":["last", "mean"],"all_activations_finite":True,
    "attention_backend":model.config._attn_implementation,
    "newly_extracted":completed,"resumed":resumed,
    "max_sequence_tokens":max(x["sequence_tokens"] for x in logs),
    "median_seconds_per_example":round(float(np.median([x["seconds"] for x in logs])),3),
    "max_peak_allocated_gib":round(max(x["peak_allocated_gib"] for x in logs),3),
    "dataset_sha256":dataset_hash,"run_key":run_key,
    "scope":"sandbox_candidate_v2_only","saved_to":str(EXTRACTION),
}
(EXTRACTION / "summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

print("SANDBOX ACTIVATION EXTRACTION VERIFIED")

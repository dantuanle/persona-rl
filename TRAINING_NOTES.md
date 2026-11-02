# Training notes (agent 2: gen_corpus, train_sft, merge_sft, train_grpo, rft_fallback, trainlib)

Checked against the tagged sources, not memory:
- TRL `v1.16.0`: `trl/trainer/{grpo_config,sft_config,base_config,grpo_trainer,sft_trainer,utils}.py`, `trl/generation/vllm_generation.py`, `trl/import_utils.py` (https://raw.githubusercontent.com/huggingface/trl/v1.16.0/...). Docs: https://huggingface.co/docs/trl/v1.16.0/en/sft_trainer, .../grpo_trainer
- transformers `v5.17.0`: `src/transformers/training_args.py`, `modeling_utils.py`, `tokenization_utils_base.py`
- peft `v0.21.2`: `src/peft/tuners/lora/config.py`, `src/peft/config.py`, `peft_model.py`, `tuners/tuners_utils.py`
- vLLM `v0.31.0`: `vllm/entrypoints/llm.py`, `vllm/sampling_params.py`, `vllm/v1/engine/parallel_sampling.py`

## Config fields checked
Every key in `config.SFT`, `config.GRPO`, `config.LORA` was checked against the dataclass fields (AST of the sources above).
`trainlib.strict_config` raises on any unknown key. Nothing is dropped.
- SFTConfig (its own fields + `_BaseConfig` + TrainingArguments): num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, lr_scheduler_type, warmup_steps, max_length (SFTConfig, default 1024), bf16, seed, save_strategy, report_to, logging_steps, max_steps: **all exist**.
- GRPOConfig: max_steps, learning_rate, lr_scheduler_type, warmup_steps, per_device_train_batch_size, gradient_accumulation_steps, num_generations, max_completion_length, temperature, beta, use_vllm, vllm_mode, vllm_gpu_memory_utilization, vllm_max_model_length, bf16, gradient_checkpointing, save_strategy, save_steps, save_only_model, logging_steps, seed, data_seed, remove_unused_columns, report_to: **all exist**.
- **`max_prompt_length` does NOT exist in GRPOConfig 1.16.0**, and the trainer never truncates prompts. `train_grpo.py` pops this one key by name, prints a loud NOTE, and enforces it: it asserts that every chat-templated GSM8K prompt is at most 512 tokens and that prompt + 512 fits in `vllm_max_model_length`=1024. The run fails if either check fails. config.py is unchanged.
- LoraConfig: r, lora_alpha, lora_dropout, bias, target_modules (LoraConfig) and task_type (PeftConfig base): **all exist**.
- Behaviour that is not obvious from config.py:
  - GRPOConfig `loss_type` defaults to `"dapo"`, `scale_rewards` to `"group"`, `vllm_importance_sampling_correction` to `True`, `num_iterations` to 1.
  - `generation_batch_size` = per_device × world × steps_per_generation, and `steps_per_generation` defaults to `gradient_accumulation_steps`. That gives 16×1×8 = 128 completions = 16 prompts × 8 per optimizer step, with one generation per step.
  - TRL `_BaseConfig` defaults `gradient_checkpointing=True`, so SFT also runs with gradient checkpointing.
- Model dtype: when given a path, both trainers load in **float32** unless `model_init_kwargs` sets dtype (documented in TRL 1.16.0). We therefore pass model instances loaded with `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`. transformers 5.17 uses `dtype` and deprecates `torch_dtype`.
- SFT loss mask: for conversational `{"prompt","completion"}` data, `completion_only_loss=None` resolves to True (`sft_trainer.py`: `self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample`). The `completion_mask` is folded into a `labels` column with -100 on the prompt tokens. `train_sft.py` asserts this on example 0 before training: the prompt part ends with `<|im_start|>assistant` and the loss covers a contiguous completion suffix.
- Rewards: a custom function is called as `reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs)`. reward_kwargs holds every dataset column except prompt/completion, plus trainer_state, log_extra and log_metric. Metric names come from `__name__`, so the format reward is renamed `format`.
- `on_log`: TRL already logs its own `step_time`. Our callback stores TRL's value as `trl_step_time` and writes the wall-clock time of the optimizer step (on_step_begin to on_step_end) as `step_time`.

## How colocate syncs PEFT weights (vllm_generation.py, `_iter_named_params` / `sync_weights`)
Yes: the adapter is merged into the base weights before `load_weights`, then unmerged.
```
with self._dist.gather_params(list(model.parameters())):
    model.merge_adapter()
    ...  name = name.removeprefix("base_model.model.").replace(".base_layer", "")
         if model.prefix in name: continue        # skip lora_A/B tensors
         yield name, param.data
    model.unmerge_adapter()
...
self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])
self.llm.reset_prefix_cache()
```
The sync runs once per optimizer step (`_last_loaded_step` guard). In colocate mode TRL builds `LLM(model=model.name_or_path, ..., distributed_executor_backend="external_launcher", seed=process_index, max_num_batched_tokens=4096, logprobs_mode="processed_logprobs")`, so vLLM loads POLICY_DIR (the B symlink works). TRL sets RANK/LOCAL_RANK/WORLD_SIZE itself and calls `ensure_master_addr_port()`. `--rendezvous` is only a fallback. TRL 1.16.0 accepts vLLM 0.22.0 to 0.31.0 (`import_utils.py`), so 0.31.0 is the newest supported version.

## Checkpoint conventions
- GRPO: step 0 = `POLICY_DIR` (no adapter; see `{GRPO_DIR}/step0_marker.json`). `checkpoint-{40,80,120}` are PEFT adapters (adapter_config.json + adapter_model.safetensors) relative to POLICY_DIR. `grpo_meta.json` lists them. The probe logs to `grpo_probe_log.jsonl`, so it never pollutes `grpo_log.jsonl`.
- RFT: `checkpoint-r{1,2,3}` are **cumulative** adapters relative to POLICY_DIR. Round r continues training the r-1 adapter, and sampling uses POLICY_DIR + adapter r-1, merged, so evals can serve them like GRPO checkpoints. The step mapping to CHECKPOINTS (40/80/120) is up to the eval side.
- Merges (`trainlib.merge_adapter_to_dir`) run on CPU in bf16. They copy every non-weight file from the base (tokenizer files, generation_config.json) and abort if one q_proj weight is unchanged after merging.

## Unresolved risks
1. Nothing here was executed on a GPU. `py_compile` passes, and the helper and filter logic was tested locally.
2. `max_prompt_length` is handled outside TRL (see above). If the lead prefers removing it from config.GRPO, the pop becomes a no-op.
3. Colocate reaches into `llm.llm_engine.model_executor.driver_worker.model_runner`. TRL claims vLLM 0.31.0 support, but with vLLM V1 that path depends on in-process execution (external_launcher). If it fails, try `--rendezvous` or fall back to `rft_fallback.py`.
4. Memory: vLLM gets 0.35 of the H100, plus a bf16 3B policy, LoRA, and gradient checkpointing at per_device 16 × 1024 tokens. Should fit, but unmeasured; run `--probe` first.
5. gen_corpus uses one SamplingParams seed for all requests (as specified), so randomness is correlated across prompts, which is harmless. It also drops neutral and rewrite outputs cut off at max_tokens, in addition to the specified filters. If fewer than N_SFT_KEEP pairs survive it warns but still writes matched corpora.
6. vLLM seeds in GRPO colocate are `process_index` (0), not config.SEED. They coincide only because SEED=0.

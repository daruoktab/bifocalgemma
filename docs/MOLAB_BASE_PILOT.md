# Molab base encoder control

Open `notebooks/pilot_e2b_base.py` in Molab and Run. Use a 96 GiB RTX PRO 6000
runtime. Supply `HF_TOKEN` or submit the notebook password form. The notebook
contains its runtime; no source release upload or Colab CLI is required.

The experiment changes the full-cross pilot encoder from E2B IT to E2B base.
Decoder IT, revisions, curriculum, ordering, seed, gate, attention and optimizer
recipe follow the shared Colab worker. Training is 100 frozen-decoder steps and
100 joint steps, not a quarter epoch. The second stage retains the original
50/50 SFT/derived curriculum. Stage seeds are 17 and 1017 respectively.

Local results are under `outputs/pilot/e2b-base`. Remote artifacts use only the
`e2b-base/` prefix of the private `daruokta/bifocalgemma-pilot-results` dataset.
Existing IT outputs and six production notebooks are not overwritten. No external
notification service is called. The notebook does not shut down the Molab VM.

Checkpoints every 25 optimizer steps and at stage ends contain FP32 trainable
weights, AdamW state and schedule groups, RNG, data cursor/order, and step history.
A checksum receipt is written last. An incomplete/corrupt checkpoint is skipped;
a different experiment identity is rejected. Run again to resume. Local work
since the last verified remote upload can be lost when the VM filesystem is lost.
`PENDING_UPLOAD` never means remotely saved. Completed runs retry upload without
loading the model. Only one local worker may own the output directory at a time.

Evaluation journals cache losses and generated cases by stage and input identity.
The eight-example gate diagnostic still performs real forwards. The recorded
`effective_gate_magnitude` is not encoder attention probability. Strict patch
scores and diagnostic hunk-recount scores remain separate. Full repair predictions
are stored alongside scores. The dev cohort has 256 examples, copy probes 16
(generation on the original first eight), and HumanEvalFix 16 cases.

`active_training_seconds` sums measured optimizer-step times represented by the
restored history; it excludes downtime and does not estimate lost uncheckpointed
steps. `wall_seconds` in a stage report describes the current invocation, not
end-to-end wall time across crashes. Timing and VRAM are platform dependent.

The frozen step-100 dev result can be compared with the available IT pilot.
Do not claim a matched joint-stage improvement unless the corresponding IT result
exists. A100 versus Molab is a platform difference, and these development probes
are not an untouched final benchmark. This pilot does not establish superiority
over the 11,102-step SFT baselines.

Regenerate with conda ai: `python tools/generate_pilot_notebook.py`.
The generator embeds the shared worker and support code. Edit those sources,
not the generated payload. A changed source identity requires explicit handling
of existing results; it cannot silently resume weights from another version.

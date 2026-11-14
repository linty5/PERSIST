# PERSIST v1.1.0

This release provides a PyTorch package for shot boundary detection. Install
from GitHub and run:

```bash
python -m pip install "git+https://github.com/linty5/PERSIST.git@v1.1.0"
persist_sbd video.mp4 -o results
```

- The release includes the paper's seed11, epoch-10 checkpoint, converted to
  safetensors without changing the model parameters. The inference threshold
  defaults to 0.5, while the validation-selected threshold of 0.68 is retained
  in the model configuration for reproducibility.
- Automatic model download and caching from [linty5/PERSIST](https://huggingface.co/linty5/PERSIST)
  on Hugging Face, fixed to `v1.1.0`, with versioned GitHub Release files as
  a fallback. Fallback files are checked by size and SHA-256.
- PyAV decoding and predictions for every frame by default. `--edges paper`
  selects the paper's inference coverage.
- Python and batch CLI interfaces, JSON shots and transitions, and CSV shot lists.
  `--per-frame` saves scores for applying a different threshold without rerunning
  the model. Frame intervals are zero-based and inclusive.

The release assets include a code-only wheel, source packages, model files and a
model ZIP. PyTorch and other dependencies are installed separately by pip. For
manual or offline use, extract the model ZIP and pass its folder to
`persist_sbd.load("./persist-model")` or `--model ./persist-model`.
See the [usage guide](https://github.com/linty5/PERSIST/blob/v1.1.0/docs/USAGE.md)
for CPU installation, options, output fields and offline loading.

The [research branch](https://github.com/linty5/PERSIST/tree/research) preserves
the training and evaluation implementation from
[v1.0.0](https://github.com/linty5/PERSIST/tree/v1.0.0). Its three original
checkpoints remain in the fixed
[v1.0.0 release](https://github.com/linty5/PERSIST/releases/tag/v1.0.0).
The [project page](https://linty5.github.io/projects/persist/) and
[BMVC 2026 paper](https://arxiv.org/abs/2608.29287) describe the method and results.

Code and weights use Apache-2.0. Upstream attributions are in `README.md`;
third-party license text is retained in `THIRD_PARTY_LICENSES/`.

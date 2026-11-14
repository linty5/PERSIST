# Using PERSIST

Start with the [quick start](../README.md#quick-start).

## Installation

Requires Python 3.10 or newer. On a CPU-only machine, install CPU PyTorch before
PERSIST to avoid downloading CUDA libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "git+https://github.com/linty5/PERSIST.git@v1.1.0"
```

The release wheel can also be installed with pip. Model files are downloaded
when first needed; pip installs runtime dependencies separately.

## Python interface

```python
import persist_sbd
model = persist_sbd.load()
result = model.detect("video.mp4")
```

Omitting `device` selects CUDA when available, otherwise CPU. Use
`persist_sbd.load(device="cpu")` or CLI `--device cpu`
to choose explicitly. To request Apple MPS, use `device="mps"`.
To process frames decoded elsewhere, pass RGB `uint8` arrays:

```python
result = model.detect_frames(frames, fps=25)
```

OpenCV returns BGR frames; convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.

## Outputs

```python
result.shots          # [(first_frame, last_frame), ...]
result.transitions    # [{"start", "end", "boundary", "kind", "peak", "run_start", "run_end"}, ...]
result.scores         # NumPy array of per-frame boundary probabilities
result.gates          # Per-frame arrays: change, transient, return, joint
result.time(frame)    # Time of a frame in seconds
result.covered        # Boolean array: whether each frame received a prediction

result.save_json("shots.json")
result.save_json("scores.json", per_frame=True)
result.save_csv("shots.csv")
stricter = result.with_threshold(0.6)               # a new result from the same scores
saved = persist_sbd.Result.load_json("scores.json")  # reload a per-frame export without the model
```

Frame indices are **zero-based and inclusive**. Shots cover the whole video.
Transition fields use the following conventions:

- `boundary`: last frame of the preceding shot, at the response interval's midpoint.
- `run_start`, `run_end`: merged response interval; one-frame gaps are joined.
- `kind`: `cut` for response spans of at most six frames, otherwise `gradual`.
- `start`, `end`: `(boundary, boundary + 1)` for cuts; the response interval for graduals.
- `peak`: maximum score in the interval.

Times are seconds relative to the first frame, using usable decoder timestamps
or frame number / frame rate (`time_source`). Unknown times are `null` in JSON
and blank in CSV. Transition `time` marks the next shot's start.

JSON records model source, revision, SHA-256, decoder, device and software versions.
`per_frame=True` also saves scores, gates, coverage and available frame timestamps.
CSV columns are `shot`, `first_frame`, `last_frame`, `start_time`, `end_time`,
`frames`, `duration`; paper mode adds `scored_frames`.
Python exports overwrite by default; pass `overwrite=False` to refuse existing files.

## Command line

```bash
persist_sbd video.mp4 -o results -f json,csv
persist_sbd videos/ -o results                      # every video in a folder
persist_sbd archive/ -r -o results --skip-existing  # with subfolders; resumes an interrupted run
persist_sbd archive/ -r -o results --per-frame      # keep the scores to re-threshold later ...
persist_sbd results/ -r --threshold 0.6 -o results_06   # ... without running the model again
```

Output names are `<video-name>.persist.json` and `<video-name>.persist.csv`;
videos from folders keep their subfolder structure under `-o`, and inputs that
would share a name keep their extension (`reel1.mov.persist.json`,
`reel1.mp4.persist.json`). Hidden files and folders are skipped. Existing outputs
require `--overwrite` or `--skip-existing`. A video that cannot be read is
reported and the others are still processed; the exit code is then 1.
After Ctrl+C, the same command with
`--skip-existing` continues where it stopped. A progress bar is shown in a
terminal (`-q` turns it off). `--selftest` checks the installation. `--download`
fetches the default model for later offline use.

## Prediction coverage and settings

| Edge mode | Coverage |
|---|---|
| `full` (default) | Score all frames using real context; repeat end frames for videos shorter than 64 frames. |
| `paper` | Score only centres of complete 64-frame windows; uncovered frames have zero scores and `covered=False`. |
| `pad` | Score all frames using repeated end frames for context. |

For example, `--edges paper` scores frames 24–103 of a 128-frame video and
scores no frames when the video is shorter than 64 frames. Empty videos are rejected.

| Python option | CLI option | Default and purpose |
|---|---|---|
| `threshold` | `--threshold` | `0.5`; finite decision threshold in `(0, 1]` |
| `edges` | `--edges` | `"full"`; `"paper"` reproduces the paper's coverage, `"pad"` fills windows with repeated end frames |
| `decoder` | `--decoder` | `"pyav"`; `"opencv"` and optional `"decord"` are also supported |
| `batch_size` | `--batch-size` | 8 on CUDA, 2 otherwise; lower it to reduce inference memory |

Pass these options to `model.detect(...)`. Batch size must be a positive integer.
For the paper's decoder, install `decord` and select `--decoder decord`.

## Local models and offline use

The default model is downloaded from [linty5/PERSIST](https://huggingface.co/linty5/PERSIST)
on Hugging Face, fixed to version `v1.1.0`. Later runs reuse cached files.
The GitHub Release copy is used automatically as a fallback, with file sizes
and SHA-256 checked before use.
To download the model before processing videos:

```bash
persist_sbd --download
```

A local model folder contains `config.json` and `model.safetensors`:

```python
model = persist_sbd.load("/path/to/model", device="cpu")
```

```bash
persist_sbd video.mp4 --model /path/to/model
```

For a manual download, extract the
[model ZIP](https://github.com/linty5/PERSIST/releases/download/v1.1.0/persist-model-v1.1.0.zip)
and pass its folder as above. Local folders do not use the network. After the
first automatic download, cached weights can also be loaded without networking:

```python
model = persist_sbd.load(local_files_only=True)
```

```bash
persist_sbd video.mp4 --local-files-only
```

Offline loading fails if the required files are missing or invalid in the
cache; it does not download them. `HF_HUB_OFFLINE=1` also disables downloads.

`load()` also accepts `cache_dir` and `revision`; the corresponding CLI flags
are `--cache-dir` and `--revision`. For example,
`persist_sbd.load(cache_dir="./model-cache")` stores downloaded files under a
chosen cache root. Leave `revision` and `variant` unset to use the default
automatic download; explicit overrides do not use the GitHub fallback.

## Research and checks

For training and paper evaluation, use the
[research implementation](https://github.com/linty5/PERSIST/tree/research).

From a source checkout, tests can use an extracted model without network access:

```bash
PERSIST_SBD_TEST_MODEL=./persist-model python -m unittest discover -s tests -v
```

## Example media

README animations use ClipShots excerpts and scores from the paper's Figure 1,
also shown on the [project page](https://linty5.github.io/projects/persist/#examples).
The underlying footage retains its original rights.

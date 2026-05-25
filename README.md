# tbexport

tbexport extracts scalars, images, and audio from TensorBoard log directories and writes them out as static HTML files with accompanying assets. Scalar tags are rendered as plot images, and image/audio tags get interactive step scrubbers.

The output is plain HTML + assets, so it works well for embedding in static site generators (Hugo, Jekyll, MkDocs, etc.) or anywhere else you want to present training results without running a TensorBoard server.

## Installation

tbexport is not yet on PyPI. To install it, clone the repo and use `uv tool install .`:

## Usage

Say you want to show off the loss, generated images, and generated audio from a recent training run in your blog. You'd run

```bash
tbexport --log-dir /path/to/tensorboard/logs \
  --scalars loss \
  --images gen_img \
  --audio gen_audio
```

Each tag produces an HTML file (e.g. `loss.html`) and an accompanying `_assets/` directory.
When media elements must be scrubbed through different type-steps, an accompanying 'scrubber.js' file is generated as well, 
which should be included in the resulting page. 

### Options

- `--log-dir` — Path to the directory containing tfevents files.
- `--scalars` — Scalar tag(s) to plot as line charts.
- `--images` — Image tag(s) to extract with step scrubbers.
- `--audio` — Audio tag(s) to extract with playback controls.

Tags can be specified individually or as a prefix to match all sub-tags. For example, `--scalars train` will match `train/loss`, `train/accuracy`, etc., and group them into a single page.

Multiple tags of the same type can be passed as separate arguments:

```bash
tbexport --log-dir ./logs --scalars loss --scalars accuracy
```

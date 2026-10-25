Unmodified copy of https://github.com/rexheng/videotesting at `eaebc5f` (`streetseg/`).

The segmentation package and its models are not edited in this repo. `run/demo.py` calls `Perception.process` on the webcam. The drone launch does not.

Weights are not in git. From this directory, once:

```bash
python -m pip install -r requirements-models.txt
python -m streetseg.download
```

Checkpoints land in `weights/`. Perception does not download them while it is running.

Unmodified copy of https://github.com/rexheng/videotesting at `eaebc5f` (`streetseg/`).

The segmentation package and its models are not edited in this repo. `run/demo.py` calls `Perception.process` on the webcam. The drone launch does not.

The checkpoints are in `weights/` and come down with a pull. Perception does not download them while it is running. To fetch them again:

```bash
python -m pip install -r requirements-models.txt
python -m streetseg.download
```

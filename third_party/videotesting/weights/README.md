# Weights

Perception loads checkpoints from this directory and does not download them
while it is running.

```bash
python3 -m pip install -r requirements-models.txt
python3 -m streetseg.download
python3 -m streetseg.download --signs --depth
```

That fetches YOLO26n-sem, YOLO26n, SegFormer-B0 trained on Mapillary Vistas
for 160k iterations, and the PVTL ResNet-18 traffic-light classifier. The
flags add the crossing-sign checkpoint and Depth Anything V2 Small. The four
checkpoints this demo uses are already in git. The Mapillary and PVTL
checkpoints are the Flying Guide Dog release,
https://github.com/EckoTan0804/flying-guide-dog.

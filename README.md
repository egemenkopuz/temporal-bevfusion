# multi-modal-3d-object-detection


## Installation

Building

```bash
. docker.sh build
```

Running

```bash
. docker.sh run-tum # on TUM server
. docker.sh run-setlabs # on Setlabs server
```

Accessing
```bash
. docker.sh exec
```

Installing
```bash
cd mmdet3d && python setup.py develop
pip install -r requirements.txt
```


## Visualization

Inside the docker container, run the following commands for headless rendering:

```bash
apt-get update && apt-get install libosmesa6-dev
pip install -r requirements-visual.txt
```

## A9 Create Data Script

```bash
git clone https://github.com/DanielPollithy/pypcd
cd pypcd && pip install .
```

```bash
python tools/create_data.py a9 --root-path ./data/a9 --out-dir ./data/a9_bevfusion
```
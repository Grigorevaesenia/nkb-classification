# NKB Classification

Easy framework for computer vision classification tasks.

## Install ...

Download from git
```bash
git clone git@github.com:nkb-tech/nkb-classification.git
cd nkb_classification
```

### ... via virtualenv

Install the package
```bash
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .
```

### ... via docker

```bash
docker build --tag nkb-cls-im --file Dockerfile .
docker run \
    -itd \
    --ipc host \
    --gpus all \
    --name nkb-cls-cont \
    --volume <host>:<docker> \ # put here path to models/configs to bind with docker image
    nkb-cls-im
docker exec -it nkb-cls-cont /bin/bash
```

## Run train

```bash
cd nkb_classification
python3 -m train -cfg `cfg_path`
```

## Run inference

```bash
cd nkb_classification
python3 -m inference -cfg `inference_cfg_path`
```

## Run onnx export

To enable export models to onnx or torchscript, run first:
```bash
python3 -m pip install -r requirements/optional.txt
```

After that, run:
```bash
cd nkb_classification
python3 -m export \
    --to onnx \  # supported [torchscript, onnx, engine]
    --config `config_model_path` \
    --opset 17 \
    --dynamic batch \  # supported [none, batch, all]
    --sim \  # simplify the graph or not
    --input-shape 1 3 224 224 \
    --device cuda:0 \
    --half \  # convert to fp16
    --save_path . \  # where to save the model
    --weights `weights_model_path`  # path to model weights
```

## Develop
To enable autocheck code before commit, run:
```bash
# install tensorrt if needed to export
python3 -m pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
python3 -m pip install -r requirements/optional.txt
pre-commit install
```

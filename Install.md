# Install.md

## Pytorch Installation

Please install Pytorch with CUDA support. You can find the installation instructions on the [Pytorch website](https://pytorch.org/get-started/locally/). 

Our program has been tested with **Python 3.10 + Pytorch 2.4.0 + CUDA 12.1**.

## Environment Setup

1. Base environment setup (details of the arguments can be found in [TRELLIS-Page](https://github.com/microsoft/TRELLIS?tab=readme-ov-file#installation-steps):
```sh
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

2. Install additional python packages:
```sh
pip install -r requirements.txt

# (for TRELLIS mesh extractor)
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VER}_cu{CUDA_VER}.html 

# (for HDR imageio read)
imageio_download_bin freeimage
```

3. Install Blender for 4D Asset Conversion:

```md
Please follow the instructions on the [Blender website](https://www.blender.org/download/) to install Blender.

Our program has been tested with **blender-4.2.1-linux-x64**.
```

4. Setup for using Hunyuan3D-2.0:
```sh
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

5. Setup for using TripoSG (TripoSG texture requires glibc 2.34 which is default in Ubuntu 22.04 (Ubuntu 20.04 need to manually install it)):

```sh
# Install Tripo-SG related models' checkpoints
cd tripoSG
python -m tripoSG.app

# (for Tripo-SG MVAdapter)
pip install spandrel==0.4.1 --no-deps 
pip install cvcuda-cu12
pip install pymeshlab==2022.2.post4
pip install tripoSG/diso-0.1.4-cp310-cp310-linux_x86_64.whl
```

6. Setup for using Craftsman3D (we use TripoSG's texture gen for Craftsman3D):
```sh
pip install lightning-utilities==0.11.2 pytorch-lightning==2.2.4 wandb typeguard open3d

cd models/checkpoints
mkdir craftsman-DoraVAE && cd craftsman-DoraVAE
wget https://pub-c7137d332b4145b6b321a6c01fcf8911.r2.dev/craftsman-DoraVAE/config.yaml
wget https://pub-c7137d332b4145b6b321a6c01fcf8911.r2.dev/craftsman-DoraVAE/model.ckpt
```
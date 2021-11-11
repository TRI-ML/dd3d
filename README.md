<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/figs/tri-logo.png" width="25%"/>
</a>

## DD3D: "Is Pseudo-Lidar needed for Monocular 3D Object detection?"

[Install](#installation) // [Datasets](#datasets) // [Experiments](#experiments) //  [Models](#models) // [License](#license) // [Reference](#reference)


<a href="https://youtu.be/rXBoUpq9CVQ" target="_blank">
<img width="100%" src="/media/figs/demo_dd3d_kitti_val_short.gif"/>
</a>

[Full video](https://youtu.be/rXBoUpq9CVQ)

Official [PyTorch](https://pytorch.org/) implementation of _DD3D_: [**Is Pseudo-Lidar needed for Monocular 3D Object detection? (ICCV 2021)**](https://arxiv.org/abs/2108.06417),
*Dennis Park<sup>\*</sup>, Rares Ambrus<sup>\*</sup>, Vitor Guizilini, Jie Li, and Adrien Gaidon*.

## Installation
We recommend using docker (see [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) instructions) to have a reproducible environment. To setup your environment, type in a terminal (only tested in Ubuntu 18.04):

```bash
git clone https://github.com/TRI-ML/dd3d.git
cd dd3d
# If you want to use docker (recommended)
make docker-build # CUDA 10.2
# Alternative docker image for cuda 11.1
# make docker-build DOCKERFILE=Dockerfile-cu111
```
Please check the version of your nvidia driver and [cuda compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) to determine which Dockerfile to use.

We will list below all commands as if run directly inside our container. To run any of the commands in a container, you can either start the container in interactive mode with `make docker-dev` to land in a shell where you can type those commands, or you can do it in one step:

```bash
# single GPU
make docker-run COMMAND="<some-command>"
# multi GPU
make docker-run-mpi COMMAND="<some-command>"
```

If you want to use features related to [AWS](https://aws.amazon.com/) (for caching the output directory)
and [Weights & Biases](https://www.wandb.com/) (for experiment management/visualization), then you should create associated accounts and configure your shell with the following environment variables **before** building the docker image:

```bash
export AWS_SECRET_ACCESS_KEY="<something>"
export AWS_ACCESS_KEY_ID="<something>"
export AWS_DEFAULT_REGION="<something>"
export WANDB_ENTITY="<something>"
export WANDB_API_KEY="<something>"
```
You should also enable these features in configuration, such as [`WANDB.ENABLED`](https://github.com/TRI-ML/dd3d/blob/main/configs/defaults.yaml#L14) and [`SYNC_OUTPUT_DIR_S3.ENABLED`](https://github.com/TRI-ML/dd3d/blob/main/configs/defaults.yaml#L29).

### Datasets
By default, datasets are assumed to be downloaded in `/data/datasets/<dataset-name>` (can be a symbolic link). The dataset root is configurable by [`DATASET_ROOT`](https://github.com/TRI-ML/dd3d/blob/main/configs/defaults.yaml#L35).

#### KITTI

The KITTI 3D dataset used in our experiments can be downloaded from the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
For convenience, we provide the standard splits used in [3DOP](https://xiaozhichen.github.io/papers/nips15chen.pdf) for training and evaluation:
```
# download a standard splits subset of KITTI
curl -s https://tri-ml-public.s3.amazonaws.com/github/dd3d/mv3d_kitti_splits.tar | sudo tar xv -C /data/datasets/KITTI3D
```

The dataset must be organized as follows:

```
<DATASET_ROOT>
    └── KITTI3D
        ├── mv3d_kitti_splits
        │   ├── test.txt
        │   ├── train.txt
        │   ├── trainval.txt
        │   └── val.txt
        ├── testing
        │   ├── calib
        |   │   ├── 000000.txt
        |   │   ├── 000001.txt
        |   │   └── ...
        │   └── image_2
        │       ├── 000000.png
        │       ├── 000001.png
        │       └── ...
        └── training
            ├── calib
            │   ├── 000000.txt
            │   ├── 000001.txt
            │   └── ...
            ├── image_2
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── label_2
                ├── 000000.txt
                ├── 000001.txt
                └── ..
```

#### nuScenes
The nuScenes dataset (v1.0) can be downloaded from the [nuScenes website](https://www.nuscenes.org/download). The dataset must be organized as follows:
```
<DATASET_ROOT>
    └── nuScenes
        ├── samples
        │   ├── CAM_FRONT
        │   │   ├── n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg
        │   │   ├── n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243512465.jpg
        │   │   ├── ...
        │   │  
        │   ├── CAM_FRONT_LEFT
        │   │   ├── n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243004917.jpg
        │   │   ├── n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243504917.jpg
        │   │   ├── ...
        │   │  
        │   ├── ...
        │  
        ├── v1.0-trainval
        │   ├── attribute.json
        │   ├── calibrated_sensor.json
        │   ├── category.json
        │   ├── ...
        │  
        ├── v1.0-test
        │   ├── attribute.json
        │   ├── calibrated_sensor.json
        │   ├── category.json
        │   ├── ...
        │  
        ├── v1.0-mini
        │   ├── attribute.json
        │   ├── calibrated_sensor.json
        │   ├── category.json
        │   ├── ...
```

### Pre-trained DD3D models
The DD3D models pre-trained on dense depth estimation using DDAD15M can be downloaded here:
| backbone | download |
| :---: | :---: |
| DLA34 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped.pth) |
| V2-99 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth) |

#### (Optional) Eigen-clean subset of KITTI raw.
To train our Pseudo-Lidar detector, we curated a new subset of KITTI (raw) dataset and use it to fine-tune its depth network. This subset can be downloaded [here](https://tri-ml-public.s3.amazonaws.com/github/dd3d/eigen_clean.txt). Each row contains left and right image pairs. The KITTI raw dataset can be download [here](http://www.cvlibs.net/datasets/kitti/raw_data.php).

### Validating installation
To validate and visualize the dataloader (including [data augmentation](./configs/defaults/augmentation.yaml)), run the following:

```bash
./scripts/visualize_dataloader.py +experiments=dd3d_kitti_dla34 SOLVER.IMS_PER_BATCH=4
```

To validate the entire training loop (including [evaluation](./configs/evaluators) and [visualization](./configs/visualizers)), run the [overfit experiment](configs/experiments/dd3d_kitti_dla34_overfit.yaml) (trained on test set):

```bash
./scripts/train.py +experiments=dd3d_kitti_dla34_overfit
```
| experiment | backbone | train mem. (GB) | train time (hr) | train log | Box AP (%) | BEV AP (%) | download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [config](configs/experiments/dd3d_kitti_dla34_overfit.yaml) | DLA-34 | 6 | 0.25 | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/dla34-kitti-overfit/logs/log.txt) | 84.54 |  88.83 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/dla34-kitti-overfit/model_final.pth) |


## Experiments
### Configuration
We use [hydra](https://hydra.cc/) to configure experiments, specifically following [this pattern](https://hydra.cc/docs/patterns/configuring_experiments) to organize and compose configurations. The experiments under [configs/experiments](./configs/experiments) describe the delta from the [default configuration](./configs/defaults.yaml), and can be run as follows:
```bash
# omit the '.yaml' extension from the experiment file.
./scripts/train.py +experiments=<experiment-file> <config-override>
```
The configuration is modularized by various components such as [datasets](./configs/train_datasets/), [backbones](./configs/backbones/), [evaluators](./configs/evaluators/), and [visualizers](./configs/visualizers), etc.


### Using multiple GPUs
The [training script](./scripts/train.py) supports (single-node) multi-GPU for training and evaluation via [mpirun](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php). This is most conveniently executed by the `make docker-run-mpi` command (see [above](#installation)).
Internally, `IMS_PER_BATCH` parameters of the [optimizer](https://github.com/TRI-ML/dd3d/blob/main/configs/common/optimizer.yaml#L5) and the [evaluator](https://github.com/TRI-ML/dd3d/blob/main/configs/common/test.yaml#L9) denote the **total** size of batch that is sharded across available GPUs while training or evaluating. They are required to be set as a multuple of available GPUs.

### Evaluation
One can run only evaluation using the pretrained models:
```bash
./scripts/train.py +experiments=<some-experiment> EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model>
# use smaller batch size for single-gpu
./scripts/train.py +experiments=<some-experiment> EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model> TEST.IMS_PER_BATCH=4
```

### Gradient accumulation
If you have insufficient GPU memory for any experiment, you can use [gradient accumulation](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa) by configuring [`ACCUMULATE_GRAD_BATCHES`](https://github.com/TRI-ML/dd3d/blob/main/configs/common/optimizer.yaml#L63), at the cost of longer training time. For instance, if the experiment requires at least 400 of GPU memory (e.g. [V2-99, KITTI](./configs/experiments/dd3d_kitti_v99.yaml)) and you have only 128 (e.g., 8 x 16G GPUs), then you can update parameters at every 4th step:
```bash
# The original batch size is 64.
./scripts/train.py +experiments=dd3d_kitti_v99 SOLVER.IMS_PER_BATCH=16 SOLVER.ACCUMULATE_GRAD_BATCHES=4
```

## Models
All experiments here use 8 A100 40G GPUs, and use gradient accumulation when more GPU memory is needed. We subsample nuScenes validation set by a factor of 8 (2Hz ⟶ 0.25Hz) to save training time.

### KITTI
| experiment | backbone | train mem. (GB) | train time (hr) | train log | Box AP (%) | BEV AP (%) | download |
| :---: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
| [config](configs/experiments/dd3d_kitti_dla34.yaml) | DLA-34 | 256 | 4.5 | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/26675chm-20210826_083148/logs/log.txt) | 16.92 |  24.77 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/26675chm-20210826_083148/model_final.pth) |
| [config](configs/experiments/dd3d_kitti_v99.yaml) | V2-99 | 400 | 9.0 | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/logs/log.txt) | 23.90 |  32.01 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/model_final.pth) |

### nuScenes
| experiment | backbone | train mem. (GB) | train time (hr) | train log | mAP (%) | NDS | download |
| :---: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
| [config](configs/experiments/dd3d_nusc_dla34.yaml) | DLA-34 | TBD | TBD | TBD) | TBD |  TBD | TBD |
| [config](configs/experiments/dd3d_nusc_v99.yaml) | V2-99 | TBD | TBD | TBD | TBD |  TBD | TBD |


## License
The source code is released under the [MIT license](LICENSE.md). We note that some code in this repository is adapted from the following repositories:
- [detectron2](https://github.com/facebookresearch/detectron2)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

## Reference
```
@inproceedings{park2021dd3d,
  author = {Dennis Park and Rares Ambrus and Vitor Guizilini and Jie Li and Adrien Gaidon},
  title = {Is Pseudo-Lidar needed for Monocular 3D Object detection?},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  primaryClass = {cs.CV},
  year = {2021},
}

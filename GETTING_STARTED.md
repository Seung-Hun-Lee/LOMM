## Getting Started with LOMM

This document provides a brief intro of the usage of LOMM.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Training
We provide a script `train_net_video.py`, that is made to train all the configs provided in LOMM.

To train a model with "train_net_video.py", first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the weights of pre-trained segmenter and put them in the current working directory.
Once these are set up, run:
```
# train the LOMM_Online
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/LOMM_Online_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/segmenter_pretrained_weights.pth \
  OUTPUT_DIR outputs/lomm_online

# train the LOMM_Offline
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/LOMM_Offline_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/online_pretrained_weights.pth \
  OUTPUT_DIR outputs/lomm_offline
```

### Evaluation

Once models are trained, run:
```
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  OUTPUT_DIR outputs/eval
```

### Reproducing Table 3-(a).
```
# (1) baseline
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.LOMM.MEMORY_MODE 'none' \
  MODEL.LOMM.OHM False \
  OUTPUT_DIR outputs/ablation/1_baseline

# (2) similarity-based memory
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.LOMM.MEMORY_MODE 'sim' \
  MODEL.LOMM.OHM False \
  OUTPUT_DIR outputs/ablation/2_sim_memory

# (3) momentum-based memory
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.LOMM.MEMORY_MODE 'momentum' \
  MODEL.LOMM.OHM False \
  OUTPUT_DIR outputs/ablation/3_momentum_memory

# (4) latest object memory (LOM)
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.LOMM.MEMORY_MODE 'lom' \
  MODEL.LOMM.OHM False \
  OUTPUT_DIR outputs/ablation/4_lom

# (5) instance occupancy memory (LOM) + occupnacy-guided Hungarian matching (OHM)
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.LOMM.MEMORY_MODE 'lom' \
  MODEL.LOMM.OHM True \
  OUTPUT_DIR outputs/ablation/5_lom_ohm

# (6) LOM + OHM + T_E
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/LOMM_Online_E_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/segmenter_pretrained_weights.pth \
  OUTPUT_DIR outputs/lomm_online_E
```


### Visualization

1. Pick a trained model and its config file. To start, you can pick from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/ovis/LOMM_Online_R50.yaml`.
2. We provide `demo_long_video.py` to visualize outputs of a trained model. Run it with:
```
python demo_long_video.py \
  --config-file /path/to/config.yaml \
  --input /path/to/images_folder \
  --output /path/to/output_folder \  
  --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth

# if the video if long (> 300 frames), plese set the 'windows_size'
python demo_long_video.py \
  --config-file /path/to/config.yaml \
  --input /path/to/images_folder \
  --output /path/to/output_folder \  
  --windows_size 300 \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth
```
The input is a folder containing video frames saved as images. For example, `ytvis_2019/valid/JPEGImages/00f88c4f0a`.

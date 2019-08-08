# Detectron with [VoVNet](https://arxiv.org/abs/1904.09730)(CVPRW'19) Backbone Networks

This repository contains [Detectron](https://github.com/facebookresearch/maskrcnn-benchmark) with [VoVNet](https://arxiv.org/abs/1904.09730) (CVPRW'19) Backbone Networks. This code based on pytorch imeplementation of Detectron ([maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)) 

## Highlights

- Memory efficient 
- Better performance, especially for *small* objects
- Faster speed


## Comparison with ResNet backbones

- 2x schedule
- same hyperparameters
- same training protocols ( max epoch, learning rate schedule, etc)
- NOT multi-scale training augmentation
- 8 x TITAN Xp GPU
- pytorch1.1
- CUDA v9
- cuDNN v7.2


| Backbone | Detector | Train mem(GB) | Inference time (ms) | Box AP (AP/APs/APm/APl) | Mask AP (AP/APs/APm/APl) | DOWNLOAD |
|----------|----------|---------------|:-------------------:|:------------------------:|:--------------------------:| :---:|
| R-50     | Faster   | 3.6           | 78                  | 37.5/21.3/40.3/49.5      | -                          |[link](https://dl.dropbox.com/s/kmcfd0j3cn9gevz/FRCN-R-50-FPN-2x.pth?dl=1)|
 | **V-39**     | Faster   | 3.9           | **78**                  | 39.8/**23.7**/42.6/51.5      | -                          |[link](https://dl.dropbox.com/s/f1per2rj4pi8t71/FRCN-V-39-FPN-2x-norm.pth?dl=1)|
 ||
 R-101    | Faster   | 4.7           | 97                  | 39.6/22.8/43.2/51.9      | -                          |[link](https://dl.dropbox.com/s/wzohk5zm9e7xw7k/FRCN-R-101-FPN-2x.pth?dl=1)|
| **V-57**     | Faster   | 4.4           | **87**                  | 40.8/**24.8**/43.8/52.4      | -                          |[link](https://dl.dropbox.com/s/rs1rgl5lupw576a/FRCN-V-57-FPN-2x-norm.pth?dl=1)|
| **V-75**     | Faster   |   5.3         |     **96**              | 41.2/**24.1**/44.3/53.0      | -                          |[link](https://dl.dropbox.com/s/311fomnmsa900l6/FRCN-V-75-FPN-2x.pth?dl=1)|
| **V-93**     | Faster   |   6.1         | 110                  |   41.8/**24.8**/45.1/53.8    | -                          |[link](https://dl.dropbox.com/s/9gcjxsf3fw1trzr/FRCN-V-93-FPN-2x.pth?dl=1)|
||
| R-50     | Mask     | 3.6           | 83                  | 38.6/22.1/41.3/51.4      | 34.9/16.0/37.3/52.2        |[link](https://dl.dropbox.com/s/dmkcu8dc662nnsu/MRCN-R-50-FPN-2x.pth?dl=1)|
| **V-39**     | Mask     | 4             | **81**                  | 41.0/**24.6**/43.9/53.1      | 36.7/**17.9**/39.3/53.0        |[link](https://dl.dropbox.com/s/3zpmq4nvijqek3m/MRCN-V-39-FPN-2x.pth?dl=1)|
||
| R-101    | Mask     | 4.7           | 102                 | 40.8/23.2/44.0/53.9      | 36.7/16.7/39.4/54.3        |[link](https://dl.dropbox.com/s/0k73qa5b8fpb45h/MRCN-R-101-FPN-2x.pth?dl=1)|
| **V-57**     | Mask     | 4.5           | **90**                  | 42.0/**25.1**/44.9/53.8      | 37.5/**18.3**/39.8/54.3        |[link](https://dl.dropbox.com/s/fawts3l0idznvvb/FRCN-V-57-FPN-2x.pth?dl=1)|
| **V-93**     | Mask   |      6.7      | 114                  | 42.7/**24.9**/45.8/55.3      | 38.0/**17.7**/40.9/55.2                          |[link](https://dl.dropbox.com/s/vf7fg36bi7nzrvf/MRCN-V-93-FPN-2x.pth?dl=1)|


## ImageNet pretrained weight

- [VoVNet-39](https://dl.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth?dl=1)
- [VoVNet-57](https://dl.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth?dl=1)
- [VoVNet-75](https://dl.dropbox.com/s/ve1h1ol2ge7yfta/vovnet75_statedict_norm.pth.tar?dl=1)
- [VoVNet-93](https://dl.dropbox.com/s/qtly316zv1isn0t/vovnet93_statedict_norm.pth.tar?dl=1)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)


## Training
Follow [the instructions](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training) [maskrcnn-benchmark](https://github.com/facebookresearch) guides

For example,

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/vovnet/e2e_faster_rcnn_V_39_FPN_2x.yaml" 
```

## Evaluation

Follow [the instruction](https://github.com/facebookresearch/maskrcnn-benchmark#evaluation)

First of all, you have to download the weight file you want to inference.

For examaple,
##### multi-gpu evaluation & test batch size 16,
```bash
wget https://dl.dropbox.com/s/f1per2rj4pi8t71/FRCN-V-39-FPN-2x-norm.pth?dl=1
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/vovnet/e2e_faster_rcnn_V_39_FPN_2x.yaml"   TEST.IMS_PER_BATCH 16 MODEL.WEIGHT FRCN-V-39-FPN-2x-norm.pth
```

##### For single-gpu evaluation & test batch size 1,
```bash
wget https://dl.dropbox.com/s/f1per2rj4pi8t71/FRCN-V-39-FPN-2x-norm.pth?dl=1
CUDA_VISIBLE_DEVICES=0
python tools/test_net.py --config-file "configs/vovnet/e2e_faster_rcnn_V_39_FPN_2x.yaml" TEST.IMS_PER_BATCH 1 MODEL.WEIGHT FRCN-V-39-FPN-2x-norm.pth
```

## TO DO LIST

 - [ ] super slim models
 - [ ] slim models
 - [ ] Larger models
 - [ ] Multi-scale training & test


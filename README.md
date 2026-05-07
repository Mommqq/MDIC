 # Distributed Image Compression with Multimodal Side Information at Extremely Low Bitrates

## Abstract

Distributed Image Compression (DIC) is crucial for multi-view transmission, especially when operating at extremely low bitrates ($<$ 0.1 bpp). Its core challenge is effectively utilizing side information to achieve high-quality reconstruction under strict bitrate budgets. However, existing DIC approaches struggle to exploit global context and object-level details from side information, leading to local blurring and the loss of fine details in the reconstruction. To address these limitations, we propose a Multimodal DIC framework (MDIC), which, for the first time, leverages side information in a multimodal manner into the DIC paradigm, effectively preserving fine-grained local details and enhancing global perceptual quality in reconstructed images. Specifically, we introduce a text-to-image diffusion-based decoder conditioned on textual side information extracted from correlated images to capture shared global semantics. Moreover, we design a feature-mask generator, supervised by a multimodal fine-grained alignment task, to strengthen the exploitation of visual side information. The generated mask serves two purposes: first, it guides the extraction of fine-grained details from losslessly transmitted side information to preserve the semantic consistency of reconstructed details; second, it regulates the extraction of clustered feature representations from the quantized VQ-VAE embeddings, compensating for category information lost under the extreme compression of the primary image. Extensive experiments on the widely used KITTI Stereo and Cityscapes datasets demonstrate that MDIC achieves state-of-the-art perceptual quality at extremely low bitrates.


## Setup
### Environment
* `Ubuntu 22.04.5 LTS`
* `Python 3.10.16`
* `PyTorch 2.3.0+cu121`

### Installation

```shell
conda create -n mdic python==3.10
conda activate mdic
pip install -r requirements.txt
```
### Dataset
The datasets used for experiments are KITTI Stereo and Cityscape.

For KITTI Stereo you can download the necessary image pairs from [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip). After obtaining `data_stereo_flow_multiview.zip` and `data_scene_flow_multiview.zip`, run the following commands:
```bash
unzip data_stereo_flow_multiview.zip # KITTI 2012
mkdir data_stereo_flow_multiview
mv training data_stereo_flow_multiview
mv testing data_stereo_flow_multiview

unzip data_scene_flow_multiview.zip # KITTI 2015
mkdir data_scene_flow_multiview
mv training data_scene_flow_multiview
mv testing data_scene_flow_multiview
```

For Cityscape you can download the image pairs from [here](https://www.cityscapes-dataset.com/downloads/). After downloading `leftImg8bit_trainvaltest.zip` and `rightImg8bit_trainvaltest.zip`, run the following commands:
```bash
mkdir cityscape_dataset
unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit cityscape_dataset
unzip rightImg8bit_trainvaltest.zip
mv rightImg8bit cityscape_dataset
```
### Train MDIC (Take Cityscapes as a instance)
```bash
 CUDA_VISIBLE_DEVICES=0 python src/train_sd_perco_2.py \
  --pretrained_model_name_or_path 'stable-diffusion-2-1' \
  --validation_frequency 5 \
  --allow_tf32 \
  --dataloader_num_workers 4 \
  --resolution 512 \
  --center_crop \
  --random_flip \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 50000 \
  --max_train_steps 500 \
  --validation_steps 500 \
  --prediction_type v_prediction \
  --checkpointing_steps 500 \
  --learning_rate 8e-05 \
  --adam_weight_decay 1e-2 \
  --max_grad_norm 1 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 10000 \
  --checkpoints_total_limit 2 \
  --dataset_name_KC Cityscape \
  --dataset_path ./cityscape_dataset \
  --output_dir PATH/result \
  --resume_from_checkpoint PATH/checkpoint.pt
```


import itertools
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import contextlib
import math
import random
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from maskpredSI import MaskedObjectPredictor
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP # type: ignore
from torchvision.transforms import ToTensor

from tqdm.auto import tqdm
from pytorch_msssim import ms_ssim
import torch.nn as nn
from packaging import version
from entroymodel import LatentTransformer

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Project-specific imports (assumed on PYTHONPATH as in your repo)
from config import ConfigMDIC as cfg_MDIC
from PairKitti import PairKitti
from PairCityscape import PairCityscape
from pipeline_sd_MDIC import StableDiffusionPipelineMDIC
from unet_2d import UNet2DConditionModel
from hyperalign import HyperEncoder, HyperEncoder_cor, CSIfusion, CommonPrior
from module import MaskGuidedFusionBlock
from helpers import prob_mask_like, get_pred_original_sample
import re
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import zlib
import torchac
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
)
# dists will be attempted when needed
from torchvision.utils import save_image
# get logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

check_min_version("0.27.0.dev0")

nlp = spacy.load("en_core_web_sm")


def mask_objects_in_sentence(sentence, vocab):

    sentence_lower = sentence.lower()
    masked_sentence = sentence
    mask_labels = []

    matches = []
    for i, word in enumerate(vocab):
        word = word.lower().strip()
        if not word:
            continue
  
        pattern = r"\b" + re.escape(word) + r"\b"
        for m in re.finditer(pattern, sentence_lower):
            matches.append((m.start(), m.end(), i, word))

    matches = sorted(matches, key=lambda x: x[0], reverse=True)

    for start, end, idx, word in matches:
        masked_sentence = masked_sentence[:start] + "[MASK]" + masked_sentence[end:]
        mask_labels.append(idx)

    return masked_sentence, mask_labels
def auto_count_objects(captions, top_k=20):
    counter = Counter()
    for cap in captions:
        doc = nlp(cap)
        for chunk in doc.noun_chunks:
            text = chunk.text.lower().strip()
            counter[text] += 1
    return counter.most_common(top_k)

def write_nested_list(f, lst, indent=2):
    prefix = " " * indent
    if isinstance(lst, (list, tuple)):
        for item in lst:
            write_nested_list(f, item, indent=indent)
    else:
        f.write(f"{prefix}{lst}\n")


def save_metrics_to_file(metrics, file_path, step=None):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        **metrics
    }
    with open(file_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def write_png(filename, image):
    image.save(filename)


def write_compressed_data_to_file(byte_stream_text, byte_stream_hyper_latent, shape, output_file):
    serialized_text = pickle.dumps(byte_stream_text)
    serialized_hyper_latent = pickle.dumps(byte_stream_hyper_latent)
    serialized_shape = pickle.dumps(shape)
    data_dict = {0: serialized_text, 1: serialized_hyper_latent, 2: serialized_shape}
    with open(output_file, "wb") as fout:
        pickle.dump(data_dict, fout)


def read_compressed_data_from_file(output_file):
    with open(output_file, "rb") as fin:
        data_dict = pickle.load(fin)
    serialized_text = data_dict[0]
    serialized_hyper_latent = data_dict[1]
    serialized_shape = data_dict[2]
    byte_stream_text = pickle.loads(serialized_text)
    byte_stream_hyper_latent = pickle.loads(serialized_hyper_latent)
    shape = pickle.loads(serialized_shape)
    return byte_stream_text, byte_stream_hyper_latent, shape


def compute_cdf_uniform_prob(codebook_size, target_shape):
    b, h, w = target_shape
    prob_per_entry = 1.0 / codebook_size
    cdf = torch.cumsum(torch.full((codebook_size,), prob_per_entry), dim=0)
    cdf = torch.cat([torch.zeros(1), cdf])
    cdf = cdf.view(1, 1, 1, -1).expand(b, h, w, -1)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    return cdf


def compress_hyper_latent(z_hat_indices):
    _, cfg_cs = cfg_MDIC.rate_cfg[cfg_MDIC.target_rate]
    cdf = compute_cdf_uniform_prob(cfg_cs, z_hat_indices.shape)
    z_hat_indices = z_hat_indices.to(torch.int16).to('cpu')
    return torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True)


def decompress_hyper_latent(compressed_hyper_latent, shape):
    cfg_ss, cfg_cs = cfg_MDIC.rate_cfg[cfg_MDIC.target_rate]
    H, W = shape
    factor = 512 // cfg_ss
    h, w = H // factor, W // factor
    cdf = compute_cdf_uniform_prob(cfg_cs, (1, int(h), int(w)))
    return torchac.decode_float_cdf(cdf, compressed_hyper_latent)


def compress_text(input_text):
    input_bytes = input_text.encode('utf-8')
    return zlib.compress(input_bytes, level=zlib.Z_BEST_COMPRESSION)


def decompress_text(compressed_text):
    decompressed_bytes = zlib.decompress(compressed_text)
    return decompressed_bytes.decode('utf-8')


def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels


# --- DDP helpers ---


def setup_ddp(ddp_flag):
    env_has = ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ)
    is_ddp = ddp_flag or env_has

    if not is_ddp:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, True, device

    # Init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Backend
    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    is_main = (rank == 0)
    return True, local_rank, world_size, is_main, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

def validate_on_val_set(
    args,
    weight_path,
    vae,
    hyperalign,
    hyperalign_cor,
    commonG,
    csifusion,
    MGF,
    EM,
    processor,
    blip2,
    text_encoder,
    tokenizer,
    unet,
    val_dataloader,
    device,
    weight_dtype,
    best_dir,
    # random_seed=42,
):

    metric_device = device
    lpips_vgg = NormFixLPIPS(net='vgg').eval().to(metric_device)

    try:
        from piq import DISTS as DISTSMetric
    except Exception as e:
        raise ImportError("Need dists-pytorch or piq to compute DISTS metric.")

    dists_metric = DISTSMetric().eval().to(metric_device)

    vae.eval()
    hyperalign.eval()
    hyperalign_cor.eval()
    unet.eval()
    text_encoder.eval()
    blip2.eval()
    commonG.eval()
    MGF.eval()
    EM.eval()


    pipeline = StableDiffusionPipelineMDIC.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        hyperalign=hyperalign,
        hyperalign_cor=hyperalign_cor,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    bpp_text_list, bpp_hyper_list = [], []
    psnr_list, msssim_list, lpips_list, dists_list, bpp_list = [], [], [], [], []
    generator = torch.Generator(device=metric_device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validation", disable=False)):
        

            img_bchw = batch['img'].to(metric_device, dtype=weight_dtype)
            B, C, H, W = img_bchw.shape
            num_pixels = H * W
            img_bchw_cor = batch['cor_img'].to(metric_device, dtype=weight_dtype)
            images_denorm = img_bchw_cor * 127.5 + 127.5
            inputs = processor(images=images_denorm, return_tensors="pt").to(metric_device)
            generated_ids = blip2.generate(**inputs, max_length=cfg_MDIC.max_number_tokens)
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions = [caption.strip() for caption in captions]

            for caption in captions:
                byte_stream_text = compress_text(caption)
                bpp_text = len(byte_stream_text) * 8 / num_pixels
                bpp_text_list.append(bpp_text)
            latents = vae.encode(img_bchw).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents.to(metric_device)
            latents_cor = vae.encode(img_bchw_cor).latent_dist.sample()
            latents_cor = latents_cor * vae.config.scaling_factor
            latents_cor = latents_cor.to(metric_device)
            hyper_latent = hyperalign(latents)
            hyper_latent_cor = hyperalign_cor(latents_cor)
            z_hat, z_hat_indices = hyper_latent.z_hat, hyper_latent.indices
            z_cor, z_hat_cor, z_hat_indices_cor = hyper_latent_cor.z, hyper_latent_cor.z_hat, hyper_latent_cor.indices          
            logits_x, prob_x = EM(z_hat_indices)
            x_seq = z_hat_indices.view(B, -1)
            nll = prob_x.gather(2, x_seq.unsqueeze(-1)).squeeze(-1) 
            total_bits = torch.sum(-torch.log(nll))
            bpp_x = total_bits / (B*H*W)
            bpp_list.append(bpp_x.cpu().item())
            
            _, cfg_cs = cfg_MDIC.rate_cfg[cfg_MDIC.target_rate]
            bpp_hyper = (z_hat_indices.numel() * math.log2(cfg_cs)) / (B * num_pixels)
            bpp_hyper_list.extend([bpp_hyper] * B)
            common_z_hat = commonG(z_hat,z_hat_cor,hard=True,thera=0.9)
            pred_mask = common_z_hat
            pred_mask = pred_mask.unsqueeze(1) 
            z_cor = MGF(pred_mask, z_cor)            
            z_hat = csifusion(z_hat_cor, z_hat, pred_mask)  
            rec_pils = pipeline(
                captions,
                z_hat,
                z_cor,
                height=H,
                width=W,
                num_inference_steps=cfg_MDIC.num_inference_steps,
                guidance_scale=cfg_MDIC.guidance_scale,
                generator=generator,
                batch_size=B
            ).images

            for j in range(B):
                rec_chw = ToTensor()(rec_pils[j]).to(metric_device)
                gt_chw = ((img_bchw[j] + 1.0) / 2.0).to(metric_device)
                mse = F.mse_loss(gt_chw, rec_chw)
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                psnr_list.append(psnr.item())

                ms_ssim_value = ms_ssim(gt_chw.unsqueeze(0), rec_chw.unsqueeze(0), data_range=1.0, size_average=True, win_size=7)
                msssim_list.append(ms_ssim_value.item())

                dists_val = dists_metric(gt_chw.unsqueeze(0), rec_chw.unsqueeze(0)).item()
                dists_list.append(dists_val)

                gt_m11 = gt_chw * 2.0 - 1.0
                rec_m11 = rec_chw * 2.0 - 1.0
                lpips_val = lpips_vgg(gt_m11.unsqueeze(0), rec_m11.unsqueeze(0), normalize=False).item()
                lpips_list.append(lpips_val)

    bpp_text_mean = np.mean(bpp_text_list) if bpp_text_list else 0.0
    bpp_hyper_mean = np.mean(bpp_hyper_list) if bpp_hyper_list else 0.0
    psnr_mean = np.mean(psnr_list) if psnr_list else 0.0
    msssim_mean = np.mean(msssim_list) if msssim_list else 0.0
    lpips_mean = np.mean(lpips_list) if lpips_list else 0.0
    dists_mean = np.mean(dists_list) if dists_list else 0.0
    bpp_mean = np.mean(bpp_list) if bpp_list else 0.0

    combined = psnr_mean - 10.0 * (lpips_mean + dists_mean)
    metrics = {
        "bpp_text": bpp_text_mean,
        "bpp_x_max": bpp_hyper_mean,
        "bpp_x_real":bpp_mean,
        "psnr": psnr_mean,
        "msssim": msssim_mean,
        "lpips": lpips_mean,
        "dists": dists_mean,
        "combined": combined,
    }
    return metrics

def unwrap_model(module):
    if isinstance(module, DDP):
        return module.module
    return module

def save_full_checkpoint(output_dir, epoch, global_step, best_combined_metric, unet, hyperalign, hyperalign_cor,commonG, csifusion, MGF, EM, maskP,optimizer, lr_scheduler, scaler, is_main_process):
    if not is_main_process:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_combined_metric": best_combined_metric,
        "unet_state_dict": unwrap_model(unet).state_dict(),
        "hyperalign_state_dict": unwrap_model(hyperalign).state_dict(),
        "hyperalign_cor_state_dict": unwrap_model(hyperalign_cor).state_dict(),
        "commonG_state_dict": unwrap_model(commonG).state_dict(),
        "csifusion_state_dict": unwrap_model(csifusion).state_dict(),
        "MGF_state_dict": unwrap_model(MGF).state_dict(),
        "EM_state_dict": unwrap_model(EM).state_dict(),
        "maskP_state_dict": unwrap_model(maskP).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    ckpt_path = os.path.join(output_dir, f"checkpoint.pt")
    torch.save(ckpt, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")
    # optionally prune older checkpoints (basic)
    # keep only latest N
    if cfg_MDIC is not None and hasattr(cfg_MDIC, "keep_ckpts") and cfg_MDIC.keep_ckpts is not None:
        pass


def load_full_checkpoint(ckpt_path, unet, hyperalign, optimizer=None, lr_scheduler=None, scaler=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    unwrap_model(unet).load_state_dict(ckpt["unet_state_dict"])
    unwrap_model(hyperalign).load_state_dict(ckpt["hyperalign_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict", None) is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


# --- main: parse args and run training ---
def parse_args():
    parser = argparse.ArgumentParser(description="MDIC training (DDP-capable)")
    # copy original args as closely as possible
    parser.add_argument(
    "--debug_disable_unet_autocast",
    action="store_true",
)

    parser.add_argument("--ddp", action="store_true", help="Enable PyTorch DDP (torchrun). If not set, single-process single-device will be used.")
    parser.add_argument("--dataset_path", default='.')
    parser.add_argument("--dataset_name_KC", default='KITTI_Stereo')
    parser.add_argument("--validation_frequency", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--train_batch_size_KC", type=int, default=1)
    parser.add_argument("--input_perturbation", type=float, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--validation_image", type=str, default=None, nargs="+")
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=" ")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_lpips", action="store_true", default=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--tracker_project_name", type=str, default=" ")
    parser.add_argument("--train_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_load", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


def main():
    args = parse_args()
    #--------kitti
    if args.dataset_name_KC == 'KITTI_Stereo' or 'KITTI_General': 
        vocab = ["a street","cars","a car","a road","trees","the street","a stop sign","the road","a building","a red car","a highway","parked cars","a path","a fence"]
    if args.dataset_name_KC == 'Cityscape': 
        vocab = ['a street','a car','cars','people','parked cars','buildings','a building','buildings','a person','a bike','trees','a bus','the street','streets']

    is_ddp, local_rank, world_size, is_main_process, device = setup_ddp(args.ddp)
    logger.info(f"DDP={is_ddp}, local_rank={local_rank}, world_size={world_size}, is_main={is_main_process}, device={device}")

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    blip2 = Blip2ForConditionalGeneration.from_pretrained(cfg_MDIC.blip_model)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, low_cpu_mem_usage=False,
        device_map=None
    )
    commonG = CommonPrior(in_channels=320)  
    csifusion = CSIfusion()
    MGF = MaskGuidedFusionBlock()
    maskP= MaskedObjectPredictor(tokenizer, text_encoder, vocab=vocab)
    unet.conv_in.weight.requires_grad = False
    unet.conv_in.bias.requires_grad = False
    nn.init.kaiming_normal_(unet.conv_in_extended.weight, a=0.2)
    nn.init.zeros_(unet.conv_in_extended.bias)    

    cfg_ss, cfg_cs = cfg_MDIC.rate_cfg[cfg_MDIC.target_rate]
    cfg_ss_cor, cfg_cs_cor = cfg_MDIC.rate_cfg_cor[cfg_MDIC.target_rate_cor]
    hyperalign = HyperEncoder(cfg_ss=cfg_ss, cfg_cs=cfg_cs)
    hyperalign_cor = HyperEncoder_cor(cfg_ss=cfg_ss_cor, cfg_cs=cfg_cs_cor)
    EM = LatentTransformer(latent_shape=(4,cfg_ss//4, cfg_ss//2), codebook_bits=int(math.log2(cfg_cs)))
    processor = Blip2Processor.from_pretrained(cfg_MDIC.blip_model)


    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    maskP.train()
    blip2.requires_grad_(False)
    unet.train()
    hyperalign.train()
    hyperalign_cor.train()
    csifusion.train()
    MGF.train()
    EM.train()

    if args.train_mask:
        commonG.train()
    else:
        commonG.requires_grad_(False)


    ema_unet = None
    if args.use_ema:
        ema_model = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None
        )
        ema_unet = EMAModel(ema_model.parameters(), model_cls=UNet2DConditionModel, model_config=ema_model.config)
        del ema_model

    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable; prefer >= 0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW
    if args.train_mask:
        trainable_parameters = list(unet.parameters()) + list(hyperalign.parameters())+list(hyperalign_cor.parameters())+list(commonG.parameters())+list(csifusion.parameters())+list(MGF.parameters())+list(EM.parameters())+list(maskP.parameters())
    else:
        trainable_parameters = list(unet.parameters()) + list(hyperalign.parameters())+list(hyperalign_cor.parameters())+list(csifusion.parameters())+list(MGF.parameters())
    optimizer = optimizer_cls(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Datasets (create on main first)
    path = args.dataset_path
    resize = (128, 256) 
    if args.dataset_name_KC == 'KITTI_Stereo':
        stereo = args.dataset_name_KC == 'KITTI_Stereo'
        train_dataset = PairKitti(path=path, set_type='train', stereo=stereo, resize=resize)
        val_dataset = PairKitti(path=path, set_type='val', stereo=stereo, resize=resize)
        test_dataset = PairKitti(path=path, set_type='test', stereo=stereo, resize=resize)
    elif args.dataset_name_KC == 'Cityscape':
        train_dataset = PairCityscape(path=path, set_type='train', resize=resize)
        val_dataset = PairCityscape(path=path, set_type='val', resize=resize)
        test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
    else:
        raise Exception("Dataset not found")

    # Distributed sampler only for train/val/test if using DDP
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_ddp else None

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True, #(train_sampler is None),
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        sampler=test_sampler,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Scheduler math
    num_warmup_steps_for_scheduler = args.lr_warmup_steps
    if args.max_train_steps is None:
        len_train = math.ceil(len(train_dataloader) / (1 if not is_ddp else 1))
        num_update_steps_per_epoch = math.ceil(len_train / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch
    else:
        num_training_steps_for_scheduler = args.max_train_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )


    weight_dtype = torch.float32
    scaler = None
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        # bf16 autocat is used without scaler

    # Move models to device
    text_encoder.to(device, dtype=weight_dtype)
    maskP.to(device)
    vae.to(device, dtype=weight_dtype)
    blip2.to(device, dtype=weight_dtype)
    unet.to(device)
    hyperalign.to(device)
    hyperalign_cor.to(device)
    commonG.to(device)
    csifusion.to(device)
    MGF.to(device)
    EM.to(device)
    # imgcortxt.to(device)
    if ema_unet is not None:
        # keep EMA on same device as unet
        ema_unet.to(device)

    lpips_model = NormFixLPIPS(net='vgg').eval().to(device)

    # Wrap trainable modules with DDP if needed (wrap unet and hyperalign)
    if is_ddp and args.train_mask:

        unet = DDP(unet, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        hyperalign = DDP(hyperalign, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        hyperalign_cor = DDP(hyperalign_cor, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        csifusion = DDP(csifusion, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        commonG = DDP(commonG, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        MGF = DDP(MGF, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        EM = DDP(EM, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
      
        
    if is_ddp and not args.train_mask:

        unet = DDP(unet, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        hyperalign = DDP(hyperalign, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        hyperalign_cor = DDP(hyperalign_cor, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        csifusion = DDP(csifusion, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
        align_feedback = DDP(align_feedback, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning("The length of the train_dataloader may not match scheduler expectations.")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if ckpt_path == "latest":
            # find latest in output_dir
            files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint") and f.endswith(".pt")]
            if files:
                files = sorted(files)
                ckpt_path = os.path.join(args.output_dir, files[-1])
        if os.path.exists(ckpt_path):
            map_location = {"cpu": "cpu"}
            if torch.cuda.is_available():
                map_location = {f"cuda:{i}": f"cuda:{local_rank}" for i in range(torch.cuda.device_count())}
            ckpt = torch.load(ckpt_path, map_location=map_location)

            if args.partial_load:
                unwrap_model(unet).load_state_dict(ckpt["unet_state_dict"])
                unwrap_model(commonG).load_state_dict(ckpt["commonG_state_dict"])
                logger.info(f"Partially loaded checkpoint {ckpt_path}, only unet + commonG. Training others from scratch.")
                global_step, first_epoch = 0, 0

            else:
                unwrap_model(unet).load_state_dict(ckpt["unet_state_dict"])
                unwrap_model(hyperalign).load_state_dict(ckpt["hyperalign_state_dict"])
                unwrap_model(hyperalign_cor).load_state_dict(ckpt["hyperalign_cor_state_dict"])
                unwrap_model(commonG).load_state_dict(ckpt["commonG_state_dict"])
                unwrap_model(csifusion).load_state_dict(ckpt["csifusion_state_dict"])
                unwrap_model(MGF).load_state_dict(ckpt["MGF_state_dict"])
                unwrap_model(EM).load_state_dict(ckpt["EM_state_dict"])
                unwrap_model(maskP).load_state_dict(ckpt["maskP_state_dict"])

                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "lr_scheduler_state_dict" in ckpt:
                    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                if "scaler_state_dict" in ckpt and scaler is not None:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])

                global_step = ckpt.get("global_step", 0)
                first_epoch = ckpt.get("epoch", 0)
                logger.info(f"Loaded checkpoint {ckpt_path}, starting from epoch {first_epoch}, step {global_step}")
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found. Starting from scratch.")


    # Training main loop
    best_combined_metric = -1e9
    sentances = []
    for epoch in range(first_epoch, args.num_train_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        epoch_progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_train_epochs}", disable=not is_main_process)
        train_loss = 0.0

        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            
            # 1) BLIP2 captions (no grad)
            images_denorm = batch['cor_img'] * 127.5 + 127.5
            B,C,H,W = images_denorm.shape
            inputs = processor(images=images_denorm, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = blip2.generate(**inputs, max_length=cfg_MDIC.max_number_tokens)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for j in range(B):
                sentances.append(generated_text[j])
            
            tokenized_captions = []
            for caption in generated_text:
                tokenized = tokenizer(caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
                tokenized_captions.append(tokenized[0])
            tokenized_captions = torch.stack(tokenized_captions).to(device)

            if cfg_MDIC.cond_drop_prob > 0.:
                prob_keep_mask = prob_mask_like((len(generated_text), 1), 1. - cfg_MDIC.cond_drop_prob, device=device)
                empty_text_embeds = torch.stack([tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0] for _ in range(tokenized_captions.shape[0])]).to(device)
                tokenized_captions = torch.where(prob_keep_mask, tokenized_captions, empty_text_embeds)
                tokenized_captions = tokenized_captions.to(device)
                

            latents = vae.encode(batch["img"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents_cor = vae.encode(batch["cor_img"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents_cor = latents_cor * vae.config.scaling_factor

            hyper_latent = hyperalign(latents)
            hyper_latent_cor = hyperalign_cor(latents_cor)
            
            z, z_hat, z_hat_indices, commit_loss = hyper_latent.z, hyper_latent.z_hat, hyper_latent.indices, hyper_latent.commit_loss
            logits_x, prob_x = EM(z_hat_indices)

            x_seq = z_hat_indices.view(B, -1)

            nll = prob_x.gather(2, x_seq.unsqueeze(-1)).squeeze(-1)  # (B, L)
            total_bits = torch.sum(-torch.log(nll))

            bpp_x = total_bits / (B*H*W)
            z_cor, z_hat_cor, z_hat_indices_cor, commit_loss_cor = hyper_latent_cor.z, hyper_latent_cor.z_hat, hyper_latent_cor.indices, hyper_latent_cor.commit_loss
                        
            pred_mask = commonG(z_hat,z_hat_cor,hard=False)

            pred_mask_hard = commonG(z_hat,z_hat_cor,hard=True,thera=0.9)
            loss_consistency = F.mse_loss(pred_mask, pred_mask_hard)
              
            masked_sentence, mask_labels = mask_objects_in_sentences(generated_text, vocab)
            loss_mask, logits_list, label_list = maskP(z_cor,masked_sentence,pred_mask,mask_labels)
     
            pred_mask = pred_mask.unsqueeze(1) 

            z_cor_init = z_cor
            z_cor = MGF(pred_mask, z_cor) 

            z_hat_init = z_hat
            z_hat = csifusion(z_hat_cor, z_hat, pred_mask) 

            bsz = latents.shape[0]

            noise = torch.randn_like(latents, device=device)
            if args.noise_offset:
                noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=device)
            if args.input_perturbation:
                new_noise = noise + args.input_perturbation * torch.randn_like(noise, device=device)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            if args.input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


            with torch.no_grad():
                encoder_hidden_states = text_encoder(tokenized_captions.to(device), return_dict=False)[0] 
            if args.prediction_type is not None:
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        
            if args.mixed_precision == "fp16":
                autocast_ctx = torch.cuda.amp.autocast
                autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)
            elif args.mixed_precision == "bf16":
                autocast_ctx = torch.autocast
                autocast_kwargs = dict(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = contextlib.nullcontext
                autocast_kwargs = {}

            with autocast_ctx(**autocast_kwargs) if autocast_ctx is not contextlib.nullcontext else contextlib.nullcontext():
      
                with torch.cuda.amp.autocast(enabled=False):
                  
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, z_hat, z_cor, return_dict=False)[0]
              
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps, device=device)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.use_lpips:
                    pred_original_sample = get_pred_original_sample(noise_scheduler, timesteps, noisy_latents, model_pred)
                    
                    latents = 1 / vae.config.scaling_factor * pred_original_sample
                    x0_pred = vae.decode(latents).sample
                    x0_pred = x0_pred.clamp(-1.0, 1.0)
                   
                    x0_pred = x0_pred.to(device) # [-1,1]
                    gt = batch['img'].to(device) # [-1,1]

                    lpips_loss = lpips_model(x0_pred, gt, normalize=False).mean()
                  
                    mse_loss = F.mse_loss(x0_pred, gt, reduction="mean")

                    loss = loss + cfg_MDIC.lambd * bpp_x + 0.1*loss_mask

            if args.train_mask:
                loss_to_backprop = loss / args.gradient_accumulation_steps
            else:
                loss_to_backprop = loss / args.gradient_accumulation_steps

            # backward (with scaler if fp16)
            if scaler is not None:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            # gradient accumulation step
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if is_ddp:
                # average loss across processes
                loss_val = loss.detach().clone()
                torch.distributed.all_reduce(loss_val, op=torch.distributed.ReduceOp.SUM)
                loss_val = loss_val.item() / (world_size if world_size > 0 else 1)
            else:
                loss_val = loss.detach().item()

            train_loss += loss_val / args.gradient_accumulation_steps
            epoch_progress.update(1)
            epoch_progress.set_postfix({"step_loss": float(loss_val), "lpips_loss": float(lpips_loss),"mse_loss": float(mse_loss), "loss_mask": float(loss_mask),"loss_consistency": float(loss_consistency),"bpp": float(bpp_x), "lr": lr_scheduler.get_last_lr()[0]})
        epoch_progress.close()
        commonG.anneal_tau()
      

        best_dir = os.path.join(args.output_dir, "best")

        if (epoch + 1) % args.validation_frequency == 0 or epoch == args.num_train_epochs-1:
            
            if is_main_process:
                val_metrics = validate_on_val_set(
                    args,
                    args.output_dir,
                    vae,
                    hyperalign if not isinstance(hyperalign, DDP) else hyperalign.module,
                    hyperalign_cor if not isinstance(hyperalign_cor, DDP) else hyperalign_cor.module,
                    commonG if not isinstance(commonG, DDP) else commonG.module,
                    csifusion if not isinstance(csifusion, DDP) else csifusion.module,
                    MGF if not isinstance(MGF, DDP) else MGF.module, 
                    EM if not isinstance(EM, DDP) else EM.module,              
                    processor,
                    blip2,
                    text_encoder,
                    tokenizer,
                    unet if not isinstance(unet, DDP) else unet.module,
                    val_dataloader,
                    device,
                    weight_dtype,
                    best_dir,
                )
                metrics_file = os.path.join(args.output_dir, "validation_metrics.jsonl")
                save_metrics_to_file(val_metrics, metrics_file, step=global_step)
                logger.info(
                    f"Validation Epoch {epoch+1} | "
                    f"BPP Text: {val_metrics['bpp_text']:.4f} | "
                    f"BPP Max: {val_metrics['bpp_x_max']:.4f} | "
                    f"BPP Real: {val_metrics['bpp_x_real']:.4f} | "
                    f"PSNR: {val_metrics['psnr']:.2f} dB | "
                    f"MS-SSIM: {val_metrics['msssim']:.4f} | "
                    f"LPIPS: {val_metrics['lpips']:.4f} | "
                    f"DISTS: {val_metrics['dists']:.4f}"
                )
                # Save best model
                if (epoch + 1) % 10== 0:
                    if val_metrics["combined"] > best_combined_metric:
                        best_combined_metric = val_metrics["combined"]
                        best_dir = os.path.join(args.output_dir, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        save_full_checkpoint(best_dir, epoch + 1, global_step, best_combined_metric, unet, hyperalign, hyperalign_cor, commonG, csifusion, MGF,EM, maskP, optimizer, lr_scheduler, scaler, is_main_process)
                        final_unet = unwrap_model(unet)
                        final_hyper = unwrap_model(hyperalign)
                        final_hyper_cor = unwrap_model(hyperalign_cor)

                        if args.use_ema and ema_unet is not None:
                            ema_unet.copy_to(final_unet.parameters())

                        pipeline = StableDiffusionPipelineMDIC.from_pretrained(
                            args.pretrained_model_name_or_path,
                            text_encoder=text_encoder,
                            vae=vae,
                            hyperalign=final_hyper,
                            hyperalign_cor=final_hyper_cor,                    
                            unet = final_unet,
                            revision=args.revision,
                            variant=args.variant,
                        )
                        pipeline.save_pretrained(best_dir)
                        logger.info(f"Saved best pipeline to {best_dir}")


if __name__ == "__main__":
    main()

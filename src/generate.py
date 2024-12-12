import os
import shutil
import tempfile
import argparse
import datetime
import json
import time
import nibabel as nib

import torch
import numpy as np
import torch.nn.functional as F
from monai import transforms


from monai.data import DataLoader
from monai.utils import first, set_determinism
from monai.metrics import PSNRMetric, SSIMMetric

from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from data_loaders_dict import get_loader

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)
tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
    """rearrange slices for buffering"""
    slices_np = np.asarray(slices)
    slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
    slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
    slices_np = slices_np[:, buffer_dim]

    _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
    b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
    x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
    if x[-1] < b_ends[-1]:
        x.append(b_ends[-1])
    n_per_batch = len(x) - 1
    windows_range = [
        range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
        for b in range(batch_size)
        for i in range(n_per_batch)
    ]
    b_slices = []
    for _s, _r in enumerate(windows_range):
        s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
        s_e = slices_np[(_r.stop - 1) % len(slices), 1]
        b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
    windows_range = itertools.chain(*windows_range)  # type: ignore
    return slices, n_per_batch, b_slices, windows_range


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)

def sliding_window_inference(
    inferer, 
    ae_sp, 
    ae_dp,
    ae_mr, 
    unet, 
    scheduler, 
    noise, 
    scale_factor,
    inputs,
    mr_images,
    roi_size,
    sw_batch_size,
    # predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.GAUSSIAN,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = True,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:

    buffered = buffer_steps is not None and buffer_steps > 0
    num_spatial_dims = len(inputs.shape) - 2
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}.")
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=not buffered)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices))
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            mr_data = mr_images[unravel_slice[0]].to(sw_device)
        if with_coord:
            # seg_prob_out = predictor(win_data, unravel_slice, *args, **kwargs)  # batched patch
            with torch.no_grad():
                sp_latent = ae_sp.encode_stage_2_inputs(win_data) * scale_factor
                mr_latent = ae_mr.encode_stage_2_inputs(mr_data) * scale_factor
                spmr_latent = torch.cat((sp_latent, mr_latent), 1)
            seg_prob_out, _ = inferer.sample(input_noise=noise, autoencoder_model=ae_dp, diffusion_model=unet, scheduler=scheduler, conditioning=spmr_latent, mode='concat', verbose=False)
            # seg_prob_out = inferer.sample(input_noise=noise, autoencoder_model=ae_dp, diffusion_model=unet, scheduler=scheduler, conditioning=sp_latent, mode='crossattn', verbose=False)
        else:
            # seg_prob_out = predictor(win_data, *args, **kwargs)  # batched patch
            with torch.no_grad():
                sp_latent = ae_sp.encode_stage_2_inputs(win_data) * scale_factor
                mr_latent = ae_mr.encode_stage_2_inputs(mr_data) * scale_factor
                spmr_latent = torch.cat((sp_latent, mr_latent), 1)
            seg_prob_out, _ = inferer.sample(input_noise=noise, autoencoder_model=ae_dp, diffusion_model=unet, scheduler=scheduler, conditioning=spmr_latent, mode='concat', verbose=False)
            # seg_prob_out = inferer.sample(input_noise=noise, autoencoder_model=ae_dp, diffusion_model=unet, scheduler=scheduler, conditioning=sp_latent, mode='crossattn', verbose=False)

        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [torch.zeros(size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device)]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][s] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                if non_blocking:
                    output_image_list[0][o_slice].copy_(sw_device_buffer[0], non_blocking=non_blocking)
                else:
                    output_image_list[0][o_slice] += sw_device_buffer[0].to(device=device)
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


parser = argparse.ArgumentParser(description='LDM-RR inferer')
parser.add_argument('-inp', '--input_folder', default='./spdp_fbp', type=str)
parser.add_argument('-out', '--output_folder', default='./results', type=str)

args = parser.parse_args()
print(args)

input_folder = args.input_folder
output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ae_sp = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)

ae_sp_path = '../checkpoints/compression_models/ae_kl_spfbp/49.ckpt'
ae_sp.load_state_dict(torch.load(ae_sp_path, map_location=device))

ae_dp = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)

ae_dp_path = '../checkpoints/compression_models/ae_kl_dpfbp/49.ckpt'
ae_dp.load_state_dict(torch.load(ae_dp_path, map_location=device))

ae_mr = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)
ae_mr_path = '../checkpoints/compression_models/ae_kl_mr/45.ckpt'
ae_mr.load_state_dict(torch.load(ae_mr_path, map_location=device))

unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=9,
    out_channels=3,
    num_res_blocks=1,
    num_channels=(32, 64, 64),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 64),
    with_conditioning=True,
    cross_attention_dim=64,
)
unet.to(device)

dmpath = '../checkpoints/latent_diffusion_model/58.ckpt'
experiment = "ldmrr"

unet.load_state_dict(torch.load(dmpath, map_location=device))

ae_sp.eval()
ae_dp.eval()
unet.eval()

scheduler_ddim = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195, clip_sample=False)
scheduler_ddim.set_timesteps(num_inference_steps=250)

dataset_val_temp = get_loader(f'./{input_folder}/val', batch_size=1, mode="train")
check_data = first(dataset_val_temp)
dp_images_temp = check_data["trgt"].to(device)
with torch.no_grad():
    with autocast(enabled=True):
        z = ae_dp.encode_stage_2_inputs(dp_images_temp)

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

inferer = LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor)

dataset_val = get_loader(f'./{input_folder}/val', batch_size=1, mode="test")

for data in dataset_val:

    sp_images = data["img"].to(device)
    mr_images = data["mr"].to(device)

    noise = torch.randn((1, 3, 16, 16, 16))
    noise = noise.to(device)

    patient_id = sp_images.meta["filename_or_obj"][0].split("/")[-1].split(".nii.gz")[0].split("_")[0]
    print(f"Working on {patient_id}...")
    if not os.path.exists(f"{output_folder}/{experiment}/{patient_id}"):
        os.makedirs(f"{output_folder}/{experiment}/{patient_id}")

    synthetic_trgt = sliding_window_inference(inferer, ae_sp, ae_dp, ae_mr, unet, scheduler_ddim, noise, scale_factor, inputs=sp_images, mr_images=mr_images, roi_size=(64, 64, 64), sw_batch_size=1)
    print(synthetic_trgt.shape)

    idx = 0
    img = synthetic_trgt[idx, 0].detach().cpu().numpy()
    print(img.shape)
    syn_image = nib.Nifti1Image(img, sp_images.meta["affine"][idx])
    nib.save(syn_image, f"{output_folder}/{experiment}/{patient_id}/syn_{patient_id}_dpFBP")

    img = sp_images[idx, :].detach().cpu().numpy().squeeze()
    sp_image = nib.Nifti1Image(img, sp_images.meta["affine"][idx])
    nib.save(sp_image, f"{output_folder}/{experiment}/{patient_id}/orig_{patient_id}_spFBP")

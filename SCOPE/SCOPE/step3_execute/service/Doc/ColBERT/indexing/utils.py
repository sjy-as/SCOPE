# import os
# import torch
# import tqdm

# from colbert.indexing.loaders import load_doclens
# from colbert.utils.utils import print_message, flatten

# def optimize_ivf(orig_ivf, orig_ivf_lengths, index_path, verbose:int=3):
#     if verbose > 1:
#         print_message("#> Optimizing IVF to store map from centroids to list of pids..")

#         print_message("#> Building the emb2pid mapping..")
#     all_doclens = load_doclens(index_path, flatten=False)

#     # assert self.num_embeddings == sum(flatten(all_doclens))

#     all_doclens = flatten(all_doclens)
#     total_num_embeddings = sum(all_doclens)

#     emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

#     """
#     EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element.
#     emb2pid_delta will have the delta from the corresponding offset,
#     """

#     offset_doclens = 0
#     for pid, dlength in enumerate(all_doclens):
#         emb2pid[offset_doclens: offset_doclens + dlength] = pid
#         offset_doclens += dlength

#     if verbose > 1:
#         print_message("len(emb2pid) =", len(emb2pid))

#     ivf = emb2pid[orig_ivf]
#     unique_pids_per_centroid = []
#     ivf_lengths = []

#     offset = 0
#     for length in tqdm.tqdm(orig_ivf_lengths.tolist()):
#         pids = torch.unique(ivf[offset:offset+length])
#         unique_pids_per_centroid.append(pids)
#         ivf_lengths.append(pids.shape[0])
#         offset += length
#     ivf = torch.cat(unique_pids_per_centroid)
#     ivf_lengths = torch.tensor(ivf_lengths)
    
#     max_stride = ivf_lengths.max().item()
#     zero = torch.zeros(1, dtype=torch.long, device=ivf_lengths.device)
#     offsets = torch.cat((zero, torch.cumsum(ivf_lengths, dim=0)))
#     inner_dims = ivf.size()[1:]

#     if offsets[-2] + max_stride > ivf.size(0):
#         padding = torch.zeros(max_stride, *inner_dims, dtype=ivf.dtype, device=ivf.device)
#         ivf = torch.cat((ivf, padding))

#     original_ivf_path = os.path.join(index_path, 'ivf.pt')
#     optimized_ivf_path = os.path.join(index_path, 'ivf.pid.pt')
#     torch.save((ivf, ivf_lengths), optimized_ivf_path)
#     if verbose > 1:
#         print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
#         if os.path.exists(original_ivf_path):
#             print_message(f"#> Original IVF at path \"{original_ivf_path}\" can now be removed")

#     return ivf, ivf_lengths

import os
import torch
import tqdm
import numpy as np

from colbert.indexing.loaders import load_doclens
from colbert.utils.utils import print_message, flatten

def optimize_ivf(orig_ivf, orig_ivf_lengths, index_path, verbose:int=3):
    stream_optimize = os.getenv('COLBERT_STREAM_OPTIMIZE_IVF', '').lower() in ('1', 'true', 'yes')
    if stream_optimize:
        return _optimize_ivf_streaming(orig_ivf, orig_ivf_lengths, index_path, verbose=verbose)

    if verbose > 1:
        print_message("#> Optimizing IVF to store map from centroids to list of pids..")

        print_message("#> Building the emb2pid mapping..")
    all_doclens = load_doclens(index_path, flatten=False)

    # assert self.num_embeddings == sum(flatten(all_doclens))

    all_doclens = flatten(all_doclens)
    total_num_embeddings = sum(all_doclens)

    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    """
    EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element.
    emb2pid_delta will have the delta from the corresponding offset,
    """

    offset_doclens = 0
    for pid, dlength in enumerate(all_doclens):
        emb2pid[offset_doclens: offset_doclens + dlength] = pid
        offset_doclens += dlength

    if verbose > 1:
        print_message("len(emb2pid) =", len(emb2pid))

    ivf = emb2pid[orig_ivf]
    unique_pids_per_centroid = []
    ivf_lengths = []

    offset = 0
    for length in tqdm.tqdm(orig_ivf_lengths.tolist()):
        pids = torch.unique(ivf[offset:offset+length])
        unique_pids_per_centroid.append(pids)
        ivf_lengths.append(pids.shape[0])
        offset += length
    ivf = torch.cat(unique_pids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)
    
    max_stride = ivf_lengths.max().item()
    zero = torch.zeros(1, dtype=torch.long, device=ivf_lengths.device)
    offsets = torch.cat((zero, torch.cumsum(ivf_lengths, dim=0)))
    inner_dims = ivf.size()[1:]

    if offsets[-2] + max_stride > ivf.size(0):
        padding = torch.zeros(max_stride, *inner_dims, dtype=ivf.dtype, device=ivf.device)
        ivf = torch.cat((ivf, padding))

    original_ivf_path = os.path.join(index_path, 'ivf.pt')
    optimized_ivf_path = os.path.join(index_path, 'ivf.pid.pt')
    torch.save((ivf, ivf_lengths), optimized_ivf_path)
    if verbose > 1:
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
        if os.path.exists(original_ivf_path):
            print_message(f"#> Original IVF at path \"{original_ivf_path}\" can now be removed")

    return ivf, ivf_lengths


def _build_emb2pid_memmap(index_path, verbose: int = 3):
    if verbose > 1:
        print_message("#> Building the emb2pid mapping (memmap)..")

    all_doclens = load_doclens(index_path, flatten=False)
    all_doclens = flatten(all_doclens)
    total_num_embeddings = int(sum(all_doclens))

    emb2pid_path = os.path.join(index_path, 'emb2pid.int32.mmap')
    emb2pid = np.memmap(emb2pid_path, dtype=np.int32, mode='w+', shape=(total_num_embeddings,))

    offset = 0
    for pid, dlength in enumerate(all_doclens):
        dlength = int(dlength)
        if dlength:
            emb2pid[offset: offset + dlength] = pid
            offset += dlength

    emb2pid.flush()
    if verbose > 1:
        print_message("len(emb2pid) =", total_num_embeddings)

    return emb2pid, total_num_embeddings


def _optimize_ivf_streaming(orig_ivf, orig_ivf_lengths, index_path, verbose: int = 3):
    if verbose > 1:
        print_message("#> Optimizing IVF (streaming) to store map from centroids to list of pids..")

    emb2pid, _ = _build_emb2pid_memmap(index_path, verbose=verbose)

    # Ensure CPU tensors.
    orig_ivf = orig_ivf.cpu()
    orig_ivf_lengths = orig_ivf_lengths.cpu().to(torch.int64)

    # Pass 1: compute unique pid counts per centroid without accumulating pids.
    if verbose > 1:
        print_message("#> Pass 1/2: computing per-centroid unique pid counts..")

    lengths_np = orig_ivf_lengths.numpy()
    ivf_pid_lengths = np.empty(len(lengths_np), dtype=np.int32)

    offset = 0
    for i, length in enumerate(tqdm.tqdm(lengths_np)):
        length = int(length)
        if length == 0:
            ivf_pid_lengths[i] = 0
            continue

        emb_ids = orig_ivf[offset: offset + length].numpy()
        pids = emb2pid[emb_ids]
        ivf_pid_lengths[i] = np.unique(pids).shape[0]
        offset += length

    total_unique = int(ivf_pid_lengths.sum())

    # Pass 2: write unique pids into a disk-backed array.
    if verbose > 1:
        print_message("#> Pass 2/2: writing optimized IVF (centroid -> unique pids)..")

    ivf_pid_path = os.path.join(index_path, 'ivf.pid.int32.mmap')
    ivf_pid = np.memmap(ivf_pid_path, dtype=np.int32, mode='w+', shape=(total_unique,))

    offset_in = 0
    offset_out = 0
    for i, length in enumerate(tqdm.tqdm(lengths_np)):
        length = int(length)
        if length == 0:
            continue

        emb_ids = orig_ivf[offset_in: offset_in + length].numpy()
        pids = emb2pid[emb_ids]
        uniq = np.unique(pids)
        out_len = uniq.shape[0]
        ivf_pid[offset_out: offset_out + out_len] = uniq
        offset_in += length
        offset_out += out_len

    ivf_pid.flush()

    optimized_ivf_path = os.path.join(index_path, 'ivf.pid.pt')
    ivf_pid_tensor = torch.from_numpy(ivf_pid)
    ivf_pid_lengths_tensor = torch.from_numpy(ivf_pid_lengths)
    torch.save((ivf_pid_tensor, ivf_pid_lengths_tensor), optimized_ivf_path)

    if verbose > 1:
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")

    return ivf_pid_tensor, ivf_pid_lengths_tensor


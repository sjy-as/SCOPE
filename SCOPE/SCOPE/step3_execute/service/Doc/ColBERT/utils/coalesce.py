import os
import argparse
import torch
from tqdm import tqdm
import ujson
import shutil
import numpy as np


def main(args):
    in_file = args.input
    out_file = args.output

    # Get num_chunks from metadata
    filepath = os.path.join(in_file, 'metadata.json')
    with open(filepath, 'r') as f:
        metadata = ujson.load(f)
    num_chunks = metadata['num_chunks']
    print(f"Num_chunks = {num_chunks}")

    # Create output dir if not already created
    if not os.path.exists(out_file):
       os.makedirs(out_file)

    if args.streaming:
        print("Coalescing doclens files (streaming)...")

        filepath = os.path.join(out_file, 'doclens.0.json')
        with open(filepath, 'w') as f:
            f.write('[')
            first = True
            for i in tqdm(range(num_chunks)):
                filepath_in = os.path.join(in_file, f'doclens.{i}.json')
                with open(filepath_in, 'r') as fin:
                    chunk = ujson.load(fin)
                for x in chunk:
                    if first:
                        first = False
                    else:
                        f.write(',')
                    f.write(str(int(x)))
            f.write(']')

        print("Coalescing codes files (streaming)...")
        num_embeddings = int(metadata['num_embeddings'])
        codes_out_path = os.path.join(out_file, '0.codes.int32')
        codes_mmap = np.memmap(codes_out_path, dtype=np.int32, mode='w+', shape=(num_embeddings,))

        offset = 0
        for i in tqdm(range(num_chunks)):
            filepath_in = os.path.join(in_file, f'{i}.codes.pt')
            chunk = torch.load(filepath_in, map_location='cpu')
            n = int(chunk.numel())
            codes_mmap[offset:offset+n] = chunk.to(torch.int32).numpy()
            offset += n
        index_len = offset
        codes_mmap.flush()
        del codes_mmap

        print("Coalescing residuals files (streaming)...")
        dim = int(metadata['config']['dim'])
        nbits = int(metadata['config']['nbits'])
        packed_dim = int(dim * nbits // 8)
        residuals_out_path = os.path.join(out_file, '0.residuals.uint8')
        residuals_mmap = np.memmap(residuals_out_path, dtype=np.uint8, mode='w+', shape=(num_embeddings, packed_dim))

        offset = 0
        for i in tqdm(range(num_chunks)):
            filepath_in = os.path.join(in_file, f'{i}.residuals.pt')
            chunk = torch.load(filepath_in, map_location='cpu')
            n = int(chunk.size(0))
            residuals_mmap[offset:offset+n, :] = chunk.to(torch.uint8).numpy()
            offset += n
        residuals_mmap.flush()
        del residuals_mmap
    else:
        ## Coalesce doclens ##
        print("Coalescing doclens files...")

        temp = []
        # read into one large list
        for i in tqdm(range(num_chunks)):
            filepath = os.path.join(in_file, f'doclens.{i}.json')
            with open(filepath, 'r') as f:
                chunk = ujson.load(f)
            temp.extend(chunk)

        # write to output json
        filepath = os.path.join(out_file, 'doclens.0.json')
        with open(filepath, 'w') as f:
            ujson.dump(temp, f)

        ## Coalesce codes ##
        print("Coalescing codes files...")

        temp = torch.empty(0, dtype=torch.int32)
        # read into one large tensor
        for i in tqdm(range(num_chunks)):
            filepath = os.path.join(in_file, f'{i}.codes.pt')
            chunk = torch.load(filepath)
            temp = torch.cat((temp, chunk))

        # save length of index
        index_len = temp.size()[0]

        # write to output tensor
        filepath = os.path.join(out_file, '0.codes.pt')
        torch.save(temp, filepath)

        ## Coalesce residuals ##
        print("Coalescing residuals files...")

        # Allocate all the memory needed in the beginning. Starting from torch.empty() and concatenating repeatedly results in excessive memory use and is much much slower.
        temp = torch.zeros(((metadata['num_embeddings'], int(metadata['config']['dim'] * metadata['config']['nbits'] // 8))), dtype=torch.uint8)
        cur_offset = 0
        # read into one large tensor
        for i in tqdm(range(num_chunks)):
            filepath = os.path.join(in_file, f'{i}.residuals.pt')
            chunk = torch.load(filepath)
            temp[cur_offset : cur_offset + len(chunk):] = chunk
            cur_offset += len(chunk)

        print("Saving residuals to output directory (this may take a few minutes)...")

        # write to output tensor
        filepath = os.path.join(out_file, '0.residuals.pt')
        torch.save(temp, filepath)

    # save metadata.json
    metadata['num_chunks'] = 1
    filepath = os.path.join(out_file, 'metadata.json')
    with open(filepath, 'w') as f:
        ujson.dump(metadata, f, indent=4)

    metadata_0 = {}
    metadata_0["num_embeddings"] = metadata["num_embeddings"]
    metadata_0["passage_offset"] = 0
    metadata_0["embedding_offset"] = 0

    filepath = os.path.join(in_file, str(num_chunks-1) + '.metadata.json')
    with open(filepath, 'r') as f:
        metadata_last = ujson.load(f)
        metadata_0["num_passages"] = int(metadata_last["num_passages"]) + int(metadata_last["passage_offset"])

    filepath = os.path.join(out_file, '0.metadata.json')
    with open(filepath, 'w') as f:
        ujson.dump(metadata_0, f, indent=4)

    filepath = os.path.join(in_file, 'plan.json')
    with open(filepath, 'r') as f:
        plan = ujson.load(f)
    plan['num_chunks'] = 1
    filepath = os.path.join(out_file, 'plan.json')
    with open(filepath, 'w') as f:
        ujson.dump(plan, f, indent=4)

    other_files = ['avg_residual.pt', 'buckets.pt', 'centroids.pt', 'ivf.pt', 'ivf.pid.pt']
    for filename in other_files:
        filepath = os.path.join(in_file, filename)
        if os.path.isfile(filepath):
            shutil.copy(filepath, out_file)

    print("Saved index to output directory {}.".format(out_file))
    print("Number of embeddings = {}".format(index_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coalesce multi-file index into a single file.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input index directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output index directory"
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Stream into raw memmap binaries to avoid high RAM usage"
    )

    args = parser.parse_args()
    main(args)

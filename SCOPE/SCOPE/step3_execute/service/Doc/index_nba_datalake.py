import os
import time


if __name__ == '__main__':
    s = time.time()

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Indexer

    base_dir = os.path.dirname(os.path.abspath(__file__))
    collection_path = os.path.join(base_dir, 'data_tsv', 'nba_datalake_title_text.tsv')
    checkpoint_path = os.path.join(base_dir, 'model_checkpoints', 'colbertv2.0')
    experiments_root = os.path.join(base_dir, 'experiments')

    if not os.path.exists(collection_path):
        raise FileNotFoundError(
            f"Collection TSV not found: {collection_path}. "
            f"Run format_tsv_nba_datalake.py first."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Download/extract ColBERTv2 checkpoint to this path."
        )

    experiment = 'nba_datalake'
    index_name = 'nba_datalake.nbits=2'

    with Run().context(RunConfig(nranks=1, experiment=experiment, avoid_fork_if_possible=True)):
        config = ColBERTConfig(
            nbits=2,
            root=experiments_root,
            avoid_fork_if_possible=True,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection_path)

    print(f"\n*** Time: {str(time.time() - s)} ***")

from mmengine.config import read_base

with read_base():
    from ..opencompass.configs.datasets.cflue.cflue_eval import cflue_datasets
    from ..opencompass.configs.models.qwen.hf_qwen3_8b_cflue import models

datasets = cflue_datasets
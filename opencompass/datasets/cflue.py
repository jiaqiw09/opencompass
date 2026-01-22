import os
import json
from datasets import Dataset, DatasetDict
from .base import BaseDataset
from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class CFLUEDataset(BaseDataset):
    """CFLUE金融领域中文语言理解评测数据集"""

    @staticmethod
    def load(path: str, file_name: str = 'knowledge/test.json', **kwargs):
        path = './data/cflue'
        dataset = DatasetDict()
        raw_data = []

        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"本地数据集文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON格式错误 in {file_path}: {str(e)}")

            for item in data:
                required_fields = ['question', 'choices', 'answer']
                for field in required_fields:
                    if field not in item:
                        raise KeyError(f"数据项缺少必要字段 '{field}': {item}")

                raw_data.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'answer': item['answer'],
                    'task': item.get('task', '')
                })
        dataset['train'] = Dataset.from_list(raw_data)
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from typing import List

class CFLUEEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        correct = 0
        total = len(predictions)
        if total != len(references):
            raise ValueError(f"预测数({total})与参考数({len(references)})不匹配")

        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip() if pred is not None else ''
            ref_clean = ref.strip() if ref is not None else ''

            pred_sorted = ''.join(sorted(pred_clean))
            ref_sorted = ''.join(sorted(ref_clean))
            if pred_sorted == ref_sorted:
                correct += 1

        accuracy = correct / total * 100.0 if total > 0 else 0.0
        return {'accuracy': round(accuracy, 4)}
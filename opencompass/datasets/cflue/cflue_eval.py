from opencompass.datasets import CFLUEDataset, CFLUEEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.utils.text_postprocessors import extract_answers

custom_imports = dict(
    imports=['opencompass.datasets.cflue'],
    allow_failed_imports=False
)
cflue_eval_cfg = dict(
    evaluator=dict(type=CFLUEEvaluator),
    pred_postprocessor=dict(type=extract_answers)
)
cflue_datasets = [
    dict(
        type=CFLUEDataset,
        path='./data/cflue',
        file_name='knowledge/test.json',
        reader_cfg=dict(
            input_columns=['question', 'choices', 'task'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='问题类型：{task}\n问题：{question}\n选项：{choices}\n 请一步步思考，并把答案选项放在 \\boxed\{\} 中，例如：答案：\\boxed\{ABC\}, 表示答案是ABC'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=cflue_eval_cfg
    )
]
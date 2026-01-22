from opencompass.models import HuggingFaceCausalLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

_meta_template = dict(
        round=[
        dict(
            role="SYSTEM",
            prompt="假设你是一位金融行业专家，请回答下列问题。",
            begin="<<|im_start|>system\n",
            end="<<|im_end|>\n",
        ),
        dict(role="HUMAN", begin="<<|im_start|>user\n", end="<<|im_end|>\n"),
        dict(
            role="BOT",
            begin="<<|im_start|>assistant\n",
            end="<<|im_end|>\n",
            generate=True,
        ),
    ],
    eos_token_id=151645
)
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="qwen3_8b_dianjin_cflue",
        path="/home/dianjin/Qwen2.5-7B-Instruct", # 待评测模型路径
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_seq_len=20480,
        max_out_len=20480,
        batch_size=2,
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
        model_kwargs=dict(
            trust_remote_code=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]

# # 单卡调试
# python run.py examples/eval_qwen25_7b_cflue_finqa.py --debug

# # 八卡并行
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py examples/eval_qwen25_7b_cflue_finqa.py --max-num-workers 8

# # vllm加速（需提前安装vllm）
# pip install ray vllm==0.11.0 vllm-ascend==0.11.0rc3
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py examples/eval_qwen25_7b_cflue_finqa.py --max-num-workers 8 -a vllm
<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> 
<p> BDoRA refers to LLM-Adapters and DoRA</p>
</h1>

<h3 align="center">
    <p>BDoRA: Balancing Magnitude and Direction in Weight-Decomposed Low-Rank Adaptation </p>
</h3>
This work is based on the LLM-Adapters integration framework, using LLaMA-7B as the base model. On the commonsense reasoning tasks of three scientific question answering datasets, we conduct systematic comparative reproduction and empirical analysis of parameter-efficient fine-tuning methods such as LoRA and DoRA, as well as experimental validation of the proposed DoRA optimization method. 

The framework provides a convenient implementation foundation for our experiments, enabling us to focus on the evaluation and improvement of method performance.

Supported PFET:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH(Series Adapter): [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel(Parallel Adapter): [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)
5. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
6. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
7. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 
8. DoRA: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arXiv.org/pdf/2402.09353v6.pdf)


## Setup
1. Install dependencies
```bash
conda create -n Bdora_llama python=3.10
conda activate Bdora_llama
pip install -r requirements.txt
```
2. Download LLaMA-7B Model
Download the LLaMA-7B model to the ./models/llama-7b-hf directory.

```bash
# Install dependency
pip install huggingface_hub

# Create target directory
mkdir -p ./models/llama-7b-hf

# Download model
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir ./models/llama-7b-hf \
    --local-dir-use-symlinks False
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and the train_scientific.json finetuning dataset, which is a mixture of the training sets from ARC-e, ARC-c, and OBQA, has already been set in this code project, then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./train_scientific.json.json
...
```
## Code Structure

Refer to `./peft/src/peft/tuners` for the implementation of all kinds of PFET include DoRA、BDoRA.

Refer to `./finetune.py` for finetuning LLaMA using all kinds of PFET besides BDoRA.

Refer to `./finetune_BDoRA.py` for finetuning LLaMA using only BDoRA.

Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model by DoRA、LoRA and BDoRA.

Refer to `./evaluate.py` for the evaluation of the finetuned model by other PFET besides LoRA and BDoRA.



## Training(finetune.py)

This file contains some code related to prompt construction and tokenization.In this file, specify different PFET and different sets of data, so that different models can be trained. 

Example usage for Single GPUs:

For LoRA:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model './models/llama-7b-hf' \
  --data_path 'train_scientific.json' \
  --output_dir './trained_models/llama-7b-lora32/' \
  --batch_size 16  --micro_batch_size 4   --num_epochs 3 \
  --learning_rate 3e-4   --cutoff_len 256   --val_set_size 120 \
  --eval_step 80 --save_step 80  --adapter_name lora \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --lora_r 32 --lora_alpha 64
```
The `train_scientific.json` data is collected with the training sets of ARC-e, ARC-c, and OBQA. `./trained_models/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.
Moreover, you can use `--use_gradient_checkpointing` to save more GPU memory, but it will increase the training time.

For Dora:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model './models/llama-7b-hf' \
    --data_path 'train_scientific.json' \
    --output_dir './trained_models/llama-7b-dora32/' \    # 32 indicates the rank is set to 32
    --batch_size 16  --micro_batch_size 16 --num_epochs 1 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name dora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing True
```

For BDora:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_BDoRA.py \
    --base_model './models/llama-7b-hf' \
    --data_path 'train_scientific.json' \
    --output_dir './trained_models/llama-7b-bdora32/' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name bdora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing True \
    --enable_adaptive_scale True \
    --bdora_lambda_reg 0.001 \
    --bdora_beta_reg 0
```

For Series Adapter:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model './models/llama-7b-hf' \
  --data_path 'train_scientific.json' \
  --output_dir './trained_models/llama-7b-bottleneck/' \
  --batch_size 16  --micro_batch_size 4   --num_epochs 3 \
  --learning_rate 3e-4   --cutoff_len 256   --val_set_size 120 \
  --eval_step 80 --save_step 80  --adapter_name bottleneck \
  --target_modules '["down_proj"]'
```

For Parallel Adapter:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model './models/llama-7b-hf' \
  --data_path 'train_scientific.json' \
  --output_dir './trained_models/llama-7b-parallel/' \
  --batch_size 16  --micro_batch_size 4   --num_epochs 3 \
  --learning_rate 3e-4   --cutoff_len 256   --val_set_size 120 \
  --eval_step 80 --save_step 80  --adapter_name bottleneck   --use_parallel_adapter \
  --target_modules '["up_proj", "down_proj"]'
```

For prefix-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model './models/llama-7b-hf' \
    --data_path 'train_scientific.json' \
    --output_dir './trained_models/llama-7b-prefix-tuning/' \
    --load_8bit True \
    --batch_size 16 --micro_batch_size 4 --num_epochs 1 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 --adapter_name prefix-tuning \
    --num_virtual_tokens 20 \
    --train_on_inputs False
```

For prompt-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model './models/llama-7b-hf' \
    --data_path 'train_scientific.json' \
    --output_dir './trained_models/llama-7b-prompt-tuning/' \
    --batch_size 16 --micro_batch_size 4 --num_epochs 1 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80    --adapter_name prompt-tuning \
    --num_virtual_tokens 20 \
    --train_on_inputs False
```

Note that, In order to facilitate INT8 training of large models with parallel adapters, we have adopted a technique whereby the parallel adapter layers are incorporated into multi-head attention layers and MLP layers, in parallel with Linear layers. It is different from [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf). 

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the commonsense Reasoning tasks, you can use the following command:

For LoRA:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset OBQA \   #dataset can be changed by ARC-c and ARC-e
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-lora32' \
```

For DoRA:
```bash
CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-dora32' \
    --batch_size 1 \
```

For DoRA:
```bash
CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-bdora32' \
    --batch_size 1 \
```

For Series Adapter:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter AdapterP \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-bottleneck' \
```

For Parallel Adapter:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter AdapterP --use_parallel_adapter \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-parallel' \
```

For prefix-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter Prefix \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-prefix-tuning' \
    --load_8bit
```

For prompt-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter Prefix \
    --dataset ARC-c \
    --base_model './models/llama-7b-hf' \
    --lora_weights './trained_models/llama-7b-prompt-tuning/' \
```


## Finetune Result
There are the finetune results in different models with 3 commonsense reasoning datasets, which contains ARC-c, ARC-e and OBQA. 

| PEFT Method | #Params (%) | ARC-c | ARC-e | OBQA | Avg |
|-------------|-------------|-------|-------|------|-----|
| Prompt-tuning | 0.001 | 54.87 | 71.26 | 64.21 | 63.44 |
| Prefix-tuning | 0.077 | 56.89 | 73.94 | 66.76 | 65.80 |
| Series Adapter | 0.996 | 60.16 | 75.56 | 74.72 | 70.14 |
| Parallel Adapter | 3.542 | 60.37 | 74.75 | 77.60 | 70.90 |
| LoRA | 0.832 | 64.59 | 78.91 | 77.20 | 73.57 |
| DoRA | 0.838 | 63.82 | 78.91 | 77.80 | 73.51 |
| **BDoRA (Ours)** | 0.838 | **64.71** | **80.12** | **77.45** | **74.09** |

## Acknowledgments & Citations

This project is built upon the following open-source frameworks:

- **LLM-Adapters** (Hu et al., 2023): [arXiv:2304.01933](https://arxiv.org/abs/2304.01933)
- **DoRA** (Liu et al., 2024): [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- **PEFT** (HuggingFace): [github.com/huggingface/peft](https://github.com/huggingface/peft)

If you use this code in your work, please consider citing these foundational papers.

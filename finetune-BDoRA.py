# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    DoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    PromptTuningConfig,  # 添加这行
    PromptTuningInit,    # 添加这行
    PromptEncoderConfig,  # 添加这行
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
'''
class BDoRATrainer(transformers.Trainer):
    def __init__(self, lambda_reg=1e-3, temperature=10.0, beta_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.beta_reg = beta_reg
        self.init_params = {}
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        task_loss = outputs.loss
        
        if self.lambda_reg > 0:
            reg_loss = 0.0
            num_layers = 0
            
            for name, module in model.named_modules():
                if hasattr(module, 'weight_m_wdecomp') and hasattr(module, 'lora_A'):
                    key = name
                    
                    m_current = module.weight_m_wdecomp.weight
                    A_current = module.lora_A.weight
                    B_current = module.lora_B.weight
                    
                    if key not in self.init_params:
                        with torch.no_grad():
                            self.init_params[key] = {
                                'm': m_current.clone().detach(),
                                'A': A_current.clone().detach(),
                                'B': B_current.clone().detach(),
                            }
                        continue
                    
                    m_init = self.init_params[key]['m']
                    A_init = self.init_params[key]['A']
                    B_init = self.init_params[key]['B']
                    
                    eps = 1e-8
                    
                    # ΔM：保留符号，添加裁剪防止爆炸
                    m_init_abs = m_init.abs() + eps
                    delta_m_signed = ((m_current - m_init) / m_init_abs).clamp(-5, 5).mean()
                    
                    # ΔD：方向变化量
                    V0 = module.weight
                    BA_current = (B_current @ A_current) * module.scaling
                    BA_init = (B_init @ A_init) * module.scaling
                    
                    V_current = V0 + BA_current
                    V_init = V0 + BA_init
                    
                    V_current_norm = V_current / (V_current.norm(dim=0, keepdim=True) + eps)
                    V_init_norm = V_init / (V_init.norm(dim=0, keepdim=True) + eps)
                    
                    cos_sim = (V_current_norm * V_init_norm).sum(dim=0).mean()
                    delta_d = (1 - cos_sim).clamp(0, 2)
                    
                    # 正相关权重
                    positive_weight = torch.sigmoid(self.temperature * delta_m_signed)
                    
                    # 正则化项（本身范围 [0, 2]）
                    product = positive_weight * delta_d
                    reg_loss += torch.relu(product - self.beta_reg)
                    num_layers += 1
            
            if num_layers > 0:
                reg_loss = reg_loss / num_layers  # 范围 [0, 2]
                # λ 通常很小 (1e-4 ~ 1e-2)，所以总损失增量很小
                total_loss = task_loss + self.lambda_reg * reg_loss
            else:
                total_loss = task_loss
        else:
            total_loss = task_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
'''
class BDoRATrainer(transformers.Trainer):
    def __init__(self, lambda_reg=1e-3, temperature=10.0, beta_reg=0.1, sample_ratio=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.beta_reg = beta_reg
        self.sample_ratio = sample_ratio
        self.init_params = {}
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        task_loss = outputs.loss
        
        if self.lambda_reg > 0:
            reg_items = []
            
            for name, module in model.named_modules():
                if hasattr(module, 'weight_m_wdecomp') and hasattr(module, 'lora_A'):
                    key = name
                    
                    m_current = module.weight_m_wdecomp.weight.view(-1)
                    A_current = module.lora_A.weight
                    B_current = module.lora_B.weight
                    
                    if key not in self.init_params:
                        with torch.no_grad():
                            self.init_params[key] = {
                                'm': m_current.clone().detach(),
                                'A': A_current.clone().detach(),
                                'B': B_current.clone().detach(),
                            }
                        continue
                    
                    m_init = self.init_params[key]['m']
                    A_init = self.init_params[key]['A']
                    B_init = self.init_params[key]['B']
                    
                    eps = 1e-8
                    
                    # ΔM（逐通道）
                    m_init_abs = m_init.abs() + eps
                    delta_m_per_channel = (m_current - m_init) / m_init_abs
                    
                    # ========== 采样计算 ΔD ==========
                    out_dim = B_current.shape[0]
                    num_samples = max(1, int(out_dim * self.sample_ratio))
                    sampled_indices = torch.randperm(out_dim)[:num_samples].to(B_current.device)
                    
                    B_sampled = B_current[sampled_indices]
                    B_init_sampled = B_init[sampled_indices]
                    
                    BA_sampled = (B_sampled @ A_current) * module.scaling
                    BA_init_sampled = (B_init_sampled @ A_init) * module.scaling
                    
                    # 余弦相似度（保留方向信息）
                    ba_norm = torch.norm(BA_sampled, dim=1) + eps
                    ba_init_norm = torch.norm(BA_init_sampled, dim=1) + eps
                    cos_sim = (BA_sampled * BA_init_sampled).sum(dim=1) / (ba_norm * ba_init_norm)
                    delta_d_mean = (1 - cos_sim).clamp(0, 2).mean()  # 标量，采样通道的平均
                    
                    # 正相关权重（逐通道）
                    positive_weight = torch.sigmoid(self.temperature * delta_m_per_channel)
                    
                    # 正则化（使用标量 delta_d_mean）
                    reg_per_channel = positive_weight * torch.relu(delta_d_mean - self.beta_reg)
                    
                    reg_items.append(reg_per_channel.mean())
                    
                    del BA_sampled, BA_init_sampled
            
            if reg_items:
                reg_loss = torch.stack(reg_items).mean()
                total_loss = task_loss + self.lambda_reg * reg_loss
            else:
                total_loss = task_loss
        else:
            total_loss = task_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
        
def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        dora_simple: bool = True,
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        #bdora
        bdora_lambda_reg: float = 0.01,  # 添加这行
        bdora_beta_reg: float = 0.01,    # 添加这行
        enable_adaptive_scale: bool = True,  # 添加这行
        bdora_temperature: float = 10.0,  # 添加这行
        # prefix tuning hyperparams
        num_virtual_tokens: int = 20,
        # prompt tuning hyperparams
        prompt_tuning_init: str = "RANDOM",  # 添加
        prompt_tuning_init_text: str = None,  # 添加
        # p-tuning hyperparams
        encoder_hidden_size: int = 128,
        encoder_num_layers: int = 2,
        encoder_dropout: float = 0.0,
        encoder_reparameterization_type: str = "MLP",
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
        f"dora_simple: {dora_simple}"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    
    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        # need to handle llama 3 separately
        if "Llama-3" in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # 修改为：
    #if load_8bit:
    #    model = prepare_model_for_int8_training(model)
#    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if load_8bit:
        model = prepare_model_for_int8_training(
            model, 
            use_gradient_checkpointing=use_gradient_checkpointing  # 保留这个参数
        )
    print(model)

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "dora":
        print("DoRA init")
        config = DoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prompt-tuning":  # 添加这个
        config = PromptTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
            prompt_tuning_init=prompt_tuning_init,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=base_model,
        )
    elif adapter_name == "p-tuning":  # 添加这个
        config = PromptEncoderConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            encoder_reparameterization_type=encoder_reparameterization_type,
        )
    elif adapter_name == "bdora":
        print("BDoRA init")
        config = DoraConfig(  # 使用原始 DoraConfig
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules,
            adaptive_scale=enable_adaptive_scale,  # 添加这行
        )
    model = get_peft_model(model, config)
        
    if load_8bit:
        model._hf_peft_config_loaded = True
        print("Marked model as PEFT-quantized model")
    # 启用梯度检查点（如果需要）
    if use_gradient_checkpointing:
        # 关闭缓存（梯度检查点必须）
        model.config.use_cache = False
        
        # 处理输入梯度要求
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # 启用梯度检查点
        if load_8bit:
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
    
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = BDoRATrainer(
        lambda_reg=bdora_lambda_reg,   # 正则化系数
        beta_reg=bdora_beta_reg,     # 边际阈值
        temperature=bdora_temperature,  # 添加这行
        sample_ratio=0.1,  # 只采样 10% 的通道
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    #model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
'''
def generate_prompt(data_point):
    # 修改为和评估一致的格式
    format_instruction = data_point["instruction"] + "\n\nPlease respond with only: the correct answer is answerX (where X is 1, 2, 3, or 4)"
    output = f"the correct answer is {data_point['output']}"
    
    if data_point.get("input"):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{format_instruction}

### Input:
{data_point['input']}

### Response:
{output}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{format_instruction}

### Response:
{output}"""
'''

if __name__ == "__main__":
    fire.Fire(train)
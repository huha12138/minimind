# 代码结构分析
# 










import warnings
import os
import torch
import gc
import time

from torch import nn
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from typing import List, Optional, Tuple  # 如果还有其他类型注解需要修正
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

import torch

def get_gpu_memory_info_torch():
    if torch.cuda.is_available():
        # 获取当前 GPU 的显存占用信息
        peak_allocated_memory = torch.cuda.max_memory_allocated() / (1024**2)
        used_memory = torch.cuda.memory_allocated() / (1024**2)  # 转换为 MB
        allocated_memory = torch.cuda.memory_reserved() / (1024**2)  # 转换为 MB
        print('\n ----------------------------------- GPU Memory Info -----------------------------------')
        print(f'Used GPU Memory : {used_memory} MB')
        print(f'Peak GPU Memory : {peak_allocated_memory} MB')
        print(f'Allocated GPU Memory : {allocated_memory} MB')
    else:
        print("CUDA is not available")

# get_gpu_memory_info_torch()

# 从基础模型（如 Mistral、Llama）中裁剪部分层，构建轻量级子模型。

# 1.前 n 层裁剪：保留前n_layers层（包括嵌入层和输出层），用于提取浅层特征
def get_first_layers_model(base_model_name: str, n_layers: int, attn_implementation: str = 'flash_attention_2'):
    """
    Builds a model comprising only the n_layers first layer of the base_model_name
    (it keeps the embedding and head layers)
    """
    full_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Create a new config for a model with fewer layers (e.g., 3 layers)
    custom_config = AutoConfig.from_pretrained(base_model_name)
    custom_config.num_hidden_layers = n_layers 
    first_layers_model = AutoModelForCausalLM.from_config(config=custom_config, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16)       
    
    # Load the state dict of the full model
    full_state_dict = full_model.state_dict()
    custom_state_dict = first_layers_model.state_dict()
    kept_state_dict = {k:v for k,v in full_state_dict.items() if k in custom_state_dict}
    
    first_layers_model.load_state_dict(kept_state_dict, strict=True)
    
    del full_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return first_layers_model

# 2.间隔层裁剪：每隔every_n_layer层保留一层，减少计算量的同时保留关键层特征。
def get_every_n_layer_model(base_model_name: str, every_n_layer: int, attn_implementation: str = 'flash_attention_2'):
    """
    Builds a model comprising 1 every every_n_layer layer of the base_model_name
    (it keeps the embedding and head layers)
    """
    full_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    n_kept_layers = full_model.config.num_hidden_layers // every_n_layer
    
    print(f'New model with 1/{every_n_layer} from {base_model_name} will have {n_kept_layers} layers')
    
    custom_config = AutoConfig.from_pretrained(base_model_name)
    custom_config.num_hidden_layers = n_kept_layers 
    custom_model = AutoModelForCausalLM.from_config(config=custom_config,   # 创建一个新的模型实例
                                                    attn_implementation=attn_implementation, 
                                                    torch_dtype=torch.bfloat16)       
    full_state_dict = full_model.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    # Filter out every Nth layer and rename to form a new state dict
    kept_state_dict = {}
    for key, value in full_state_dict.items():
        if ".layers." in key:
            # Extract layer index
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            # Check if it's an Nth layer
            if layer_idx % every_n_layer == 0:
                # Adjust layer index for the smaller model
                new_layer_idx = layer_idx // every_n_layer
                # print('replacing', f".layers.{layer_idx}.", f".layers.{new_layer_idx}.")
                new_key = key.replace(f".layers.{layer_idx}.", f".layers.{new_layer_idx}.")
                if new_key in custom_state_dict:
                    kept_state_dict[new_key] = value
        else:
            # Keep non-layer-specific parameters
            if key in custom_state_dict:
                kept_state_dict[key] = value

    # Load the filtered state dict into the custom model
    custom_model.load_state_dict(kept_state_dict, strict=True)
    
    del full_model  # 删除对完整模型full_model的引用，使其成为垃圾对象，等待垃圾回收。
    torch.cuda.empty_cache()    # 清空 PyTorch 的 CUDA 缓存，释放未被使用但仍占用显存的内存块。
    gc.collect()    # 显式调用 Python 的垃圾回收机制，清理所有未被引用的对象（包括full_model及其参数张量）。
    
    return custom_model
                
# 构建一个经过裁剪的基础模型，封装压缩模型
class MistralTrimmed(torch.nn.Module):
    """
    Trimmed version of base models for faster compression
    NB: the name 'MistralTrimmed' suggests it just works with mistral but NO in fact most LLMs are supported !
    """
    def __init__(self, 
                 n_layers: int = 15,
                 every_n_layer: int = None,
                 rms_norm: bool = False,
                 base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 attn_implementation: str = 'flash_attention_2'):
        """
        you can either specify
        - n_layers to some number: we take the n_layers first layers of the base model.
        - every_n_layer to some number: in that case we take 1/N layer of the base model
        The base_model_name is the name of the model from which this model is built.
        """
        assert (n_layers is None) ^ (every_n_layer is None), 'Cannot specify both n_layers and every_n_layer for MistralTrimmed'        
        super().__init__()
        
        self.n_layers = n_layers
        self.every_n_layer = every_n_layer
        self.base_model_name = base_model_name
        
        if n_layers is not None:
            self.custom_model = get_first_layers_model(self.base_model_name, 
                                                       n_layers, 
                                                       attn_implementation=attn_implementation)
        
        else:
            self.custom_model = get_every_n_layer_model(self.base_model_name, 
                                                        every_n_layer, 
                                                        attn_implementation=attn_implementation)
        # 将模型的数据类型设置为 bfloat16，并将模型移动到 GPU 上进行计算，以提高计算效率
        self.custom_model = self.custom_model.bfloat16()
        self.custom_model.cuda()

        if rms_norm:
            print('Compressor keeps its original rms norm')
        else:
            print('De-activating RMS norm in compressor')
            # We deactivate the norm: we don't need it here since we want to manipulate stuff within embed space
            # see https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/models/mistral/modeling_mistral.py#L699
            self.custom_model.model.norm = nn.Identity()
        
        # Piping useful methods:    方法封装：将基础模型的一些有用方法封装到 MistralTrimmed 类中，方便调用。
        self.add_adapter = self.custom_model.add_adapter
        self.set_adapter = self.custom_model.set_adapter
        self.load_adapter = self.custom_model.load_adapter
        self.num_parameters = self.custom_model.num_parameters
        self.resize_token_embeddings = self.custom_model.resize_token_embeddings
        self.get_input_embeddings = self.custom_model.get_input_embeddings
        self.get_adapter_state_dict = self.custom_model.get_adapter_state_dict
        
        # self.custom_model.gradient_checkpointing_enable()
        
        # del self.custom_model.lm_head # THIS FAILS since some models have tie_embeddings=True !
        # gc.collect()
        # torch.cuda.empty_cache()    

    def forward(self, input_ids, attention_mask=None):
    # 输入 词id，转化为对应嵌入向量表示，然后进行注意力操作与FFN ；output_hidden_states=True，模型会返回每一层的隐藏状态
    # input_ids 是一个整数列表（在 PyTorch 中通常表现为 torch.Tensor 类型），它是文本经过分词（tokenization）后，每个词元（token）对应于词汇表（vocabulary）中索引的数值表示。
    # 简单来说，它是将文本从字符串形式转换为模型能够处理的数字形式。

    # 通过调用 self.custom_model.model 而不是 self.custom_model，可以避免计算语言模型头（LM head）的开销。语言模型头通常是一个线性层，用于将最后一层的隐藏状态映射到词汇表大小的向量，以便进行下一个词的预测。
    # 在某些情况下，我们可能只需要模型的隐藏状态，而不需要进行词预测，因此可以跳过语言模型头的计算，提高计算效率。

        return self.custom_model.model(input_ids, attention_mask, output_hidden_states=True) # we call the .model attribute of the causal LM to avoid the cost of the LM head ! nice huh ?

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        return self.forward(input_ids, attention_mask)


class AbstractCompressor(nn.Module):
    def __init__(self, compr_model_name: str, compr_rate: int, decoder_hidden_size: int):
        super().__init__()
        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.decoder_hidden_size = decoder_hidden_size
        
    def forward(self, input_ids, attention_mask, generation_top_k):
        """
        input_ids of shape (batch_size, top_k, seq_length)
        attention_mask of shape (batch_size, top_k, seq_length)
        generation_top_k: the number of docs  Q
        """
        raise NotImplementedError
    
    def save_pretrained(self, save_directory):
        raise NotImplementedError

    def load_pretrained(self, load_directory):
        raise NotImplementedError


class BertCompressor(AbstractCompressor):
    def __init__(self, 
                 compr_model_name: str,
                 compr_rate: int,   # 压缩率
                 decoder_hidden_size: int, 
                 mlp_hidden_dim: int = 8192, 
                 use_mlp: bool = True,
                 doc_max_length : int = 128,
                 **kwargs):
        # TODO use the device_map
        super().__init__(compr_model_name=compr_model_name, compr_rate=compr_rate, decoder_hidden_size=decoder_hidden_size)
        if compr_model_name == 'mistral_trimmed':   # null
            assert 'compr_n_layers' in kwargs
            self.model = MistralTrimmed(n_layers=kwargs['compr_n_layers'], 
                                        every_n_layer=kwargs['compr_every_n_layer'], 
                                        rms_norm=kwargs['compr_rms_norm'],
                                        base_model_name=kwargs['compr_base_model_name'],
                                        attn_implementation=kwargs['attn_implementation'])
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.base_model_name)
            self.hidden_size = self.model.custom_model.config.hidden_size
        else:
            print('-------------- 加载压缩模型 ---------------------------')
            self.model = AutoModel.from_pretrained(compr_model_name, torch_dtype=torch.bfloat16, device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(compr_model_name, use_fast=True)
            self.tokenizer.padding_side = "left"
            self.hidden_size = self.model.config.hidden_size
        
        print("compressor model name:",compr_model_name)    # model/llm/Mistral-7B-Instruct-v0.2
        print('Base compressor nb parameters', self.model.num_parameters()) # 输出压缩器的参数大小

        self.mlp_hidden_dim = mlp_hidden_dim
        self.use_mlp = use_mlp
        self.doc_max_length = doc_max_length    # 限制文档最大长度 Q：真的限制了吗？
        
        if self.use_mlp:    # Q：mlp的作用是？ 嵌入投影
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.mlp_hidden_dim), 
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_dim, decoder_hidden_size)
            ).bfloat16() 
            self.mlp.cuda()
        
        self.n_emb = self.doc_max_length // self.compr_rate # 特殊压缩token的数量
        
        mem_tokens = ['<MEM' + str(i) + '>' for i in range(self.n_emb)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': mem_tokens}) 
        self.tokenizer.mem_tokens = mem_tokens
        self.tokenizer.mem_token_ids = [self.tokenizer.convert_tokens_to_ids(elt) for elt in self.tokenizer.mem_tokens]
        self.tokenizer.mem_token_ids_pt = torch.LongTensor(self.tokenizer.mem_token_ids)
        self.model.resize_token_embeddings(len(self.tokenizer))
            
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            
        if not use_mlp:
            assert decoder_hidden_size == self.hidden_size, f'Mlp mandatory is hidden sizes not equal: {decoder_hidden_size} vs {self.hidden_size}'
            
        self.lora = False
        self.lora_name = 'compr_adapter'
        
    def prepare_mem_tokens_optimization(self):
        """此方法的主要目的是对模型中词嵌入层的梯度进行特定优化，让模型在训练期间仅对特殊的记忆标记（mem tokens）对应的词嵌入向量进行更新，
        而其他词嵌入向量则保持不变。该方法通常会在使用 LoRA（Low-Rank Adaptation）技术进行模型微调时被调用。"""

        assert self.lora, 'should only be called with lora.'
        self.model.get_input_embeddings().weight.requires_grad = True
        # Applying a hook zero-ing the gradients except for the mem token:
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[self.tokenizer.mem_token_ids] = 1.0
            return grad * mask
        self.model.get_input_embeddings().weight.register_hook(hook)            
        
    def set_lora(self, peft_config):
        self.model.add_adapter(peft_config, self.lora_name)
        self.model.set_adapter(self.lora_name)
        self.lora = True
        self.prepare_mem_tokens_optimization()

    def forward(self, input_ids, attention_mask):
        """核心功能是从输入序列中提取特定记忆标记（mem tokens）位置的隐藏状态，并根据配置对这些隐藏状态进行处理，最后返回处理后的结果。"""
        
        assert input_ids.size() == attention_mask.size()
        assert len(input_ids.size()) == 2
        
        batch_size_times_top_k = input_ids.size(0)  # Q:topk指的是多个文档吗？
        
        last_hidden_states = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        output_hidden_states=True).hidden_states[-1]
       
        # 将LLM压缩器的最后一层隐藏层数据平均化即为COCOM设计
        # Getting the hidden states at the mem token positions, as for regular cocom:
        mask = torch.isin(input_ids, self.tokenizer.mem_token_ids_pt.to(input_ids.device))
        selected_n_tokens = last_hidden_states[mask].reshape(last_hidden_states.size(0), -1, last_hidden_states.size(-1))            
        # 提取特殊token的隐藏状态

        assert selected_n_tokens.size() == (batch_size_times_top_k, self.n_emb, self.hidden_size), f"{selected_n_tokens.size()} vs {(batch_size_times_top_k, self.n_emb, self.hidden_size)}"
        
        if self.use_mlp:    # Q：所以实际上是将文档信息压缩在多个特殊token中，然后使用mlp转换下？
            selected_n_tokens = self.mlp(selected_n_tokens) # now of shape (batch_size, top_k, decoder_hidden_size) # Q：变成单向量表示了吗？
            
        assert selected_n_tokens.size() == (batch_size_times_top_k, self.n_emb, self.decoder_hidden_size), f"{selected_n_tokens.size()} vs {(batch_size_times_top_k, self.n_emb, self.decoder_hidden_size)}"
        
        return selected_n_tokens
    
    # 拼接本地路径和模型名称，本地加载模型地址
    def get_lora_path_from_directory(self, directory):
        print("本地加载压缩器微调模型 load local compressor-adapter")
        return os.path.join(directory, 'compressor_adapters.pth')
    
    def get_compressor_path_from_directory(self, directory):
        print("本地加载压缩器 load local compressor")
        return os.path.join(directory, 'compressor.pth')
    
    def get_mlp_path_from_directory(self, directory):
        print("本地加载MLP load local mlp")
        return os.path.join(directory, 'mlp.pth')
    
    def get_first_layer_path_from_directory(self, directory):
        print("本地加载第一层模型 load local first layer")
        return os.path.join(directory, 'first_layer.pth')
    
    def get_first_layer_state_dict(self) -> dict:
        # 从模型的所有参数里找出词嵌入层（embed_tokens）的权重参数，并将其转换为 CPU 上的张量，最后以字典的形式返回。
        out = {}
        for k, v in self.model.named_parameters():
            if 'embed_tokens.weight' in k:
                out[k] = v.cpu()
        # 将张量转换为 CPU 上的张量的核心目的是 提高状态字典的通用性、兼容性和设备无关性，确保模型参数可在任意环境中保存、加载和使用，同时避免设备依赖导致的错误。这是 PyTorch 中处理模型参数时的常见最佳实践，尤其在涉及模型存储、迁移和跨设备部署时至关重要。       
        assert len(out) == 1, len(out) # We should get exactly one layer here
        return out
    
    def save_pretrained(self, save_directory):
        """
        Here we just save mlp state_dict and model state_dict
        Config is handled in cocom model.   # 根据参数设置保存模型权重
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save MLP weights
        if self.use_mlp:
            mlp_path = self.get_mlp_path_from_directory(directory=save_directory)
            torch.save(self.mlp.state_dict(), mlp_path)
        
        # Saving the model
        if not self.lora: # full training: save the full dict:
            model_path = self.get_compressor_path_from_directory(directory=save_directory)
            torch.save(self.model.state_dict(), model_path)
        else: # lora training of the compressor # 似乎适用于7B模型
            # We save the first layer:
            first_layer_state_dict = self.get_first_layer_state_dict()
            torch.save(first_layer_state_dict, self.get_first_layer_path_from_directory(directory=save_directory))
            
            # We save the adapters:
            adapter_state_dict = {k: v.cpu() for k, v in self.model.get_adapter_state_dict(self.lora_name).items()}
            torch.save(adapter_state_dict, self.get_lora_path_from_directory(directory=save_directory))
            
    def load_adapter(self, load_directory, peft_config):    
        assert peft_config is not None
        map_location = torch.device("cpu") if not torch.cuda.is_available else None
        adapter_state_dict = torch.load(self.get_lora_path_from_directory(directory=load_directory), map_location=map_location, weights_only=True)
        print('loading compr adapter onto compressor model from', self.get_lora_path_from_directory(directory=load_directory))
        self.model.load_adapter(peft_config=peft_config, adapter_name=self.lora_name, adapter_state_dict=adapter_state_dict)
        self.lora = True
        self.prepare_mem_tokens_optimization()
        
    def load_first_layer(self, load_directory):
        map_location = torch.device("cpu") if not torch.cuda.is_available else None
        first_layer_state_dict = torch.load(self.get_first_layer_path_from_directory(load_directory), map_location=map_location, weights_only=True)
        assert len(first_layer_state_dict.keys()) == 1
        self.model.load_state_dict(first_layer_state_dict, strict=False)

    def load_pretrained(self, load_directory, lora: bool = False, peft_config=None):
        """
        Loading the state dicts.
        :lora: if True then the compressor was trained using lora: we just need to load the adapters
        if False, the compressor was fully trained: we load it fully.
        """
        if self.use_mlp:
            mlp_path = self.get_mlp_path_from_directory(directory=load_directory)
            self.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
        
        if lora:
            self.load_first_layer(load_directory)
            self.load_adapter(load_directory, peft_config)
        else:
            model_path = self.get_compressor_path_from_directory(directory=load_directory)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))  
            
    def prepare_inputs(self, texts, max_length, q_texts=None):
        # 采用不同的编码策略对输入文本进行编码，并为编码后的输入添加记忆标记。这样处理后的数据（input_ids和attention_mask）可以直接用于模型的前向传播。
        if q_texts is not None: # Query-dependent here: # Q：查询相关的压缩？
            assert len(texts) == len(q_texts), f"{len(texts)} == {len(q_texts)}"
            if self.compr_model_name == 'mistral_trimmed':
                # No special token, just formulating:
                texts_to_encode = [ '\nQuery:\n' + query + 'Document:\n' + text for text, query in zip(texts, q_texts)]
                inp_enc = self.tokenizer(texts_to_encode, 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        max_length=max_length + 8, # some margin for query/doc stuff + bos / eos
                                        truncation=True,
                                        add_special_tokens=True)
            else:
                inp_enc = self.tokenizer(q_texts,  # we put the query in first position
                                        texts, 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        max_length=max_length + 3,
                                        truncation='only_second',
                                        add_special_tokens=True)
        else:
            inp_enc = self.tokenizer(texts, return_tensors='pt', padding='max_length', max_length=max_length + 2, truncation=True, add_special_tokens=True)
            
        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'], 
                                                                                        inp_enc['attention_mask'], 
                                                                                        self.n_emb, 
                                                                                        tokenizer=self.tokenizer)
            
        return inp_enc

# class ending
################################################################################################################################################################################
# 将一定数量的记忆标记（memory tokens）拼接到输入的 input_ids 之后，并且相应地更新 attention_mask，以确保模型在处理输入时能够正确关注这些新添加的记忆标记。
# 这样经过LLM计算之后将文档信息压缩在特殊token之中
def add_memory_tokens_to_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, n_mem_tokens: int, tokenizer):
    """
    Concatenate the input ids with n_mem_tokens mem_tokens and update the corresponding attention mask
    input_ids: token对应的整数序列
    """
    assert len(tokenizer.mem_tokens) == n_mem_tokens, f"{len(tokenizer.mem_tokens)} VS {n_mem_tokens}"

    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens.size()) == 2
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens
    #mem_tokens = torch.full((input_ids.size(0), n_mem_tokens), tokenizer.mem_token_id, dtype=torch.long)
    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), n_mem_tokens)], dim=1)
    return input_ids, attention_mask


class COCOMConfig(PretrainedConfig):

    model_type = "COCOM"
    def __init__(self,
                decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                doc_max_length: int = 128,
                quantization: str = 'no',
                sep: bool = False,
                compr_model_name: str = "google-bert/bert-base-uncased",
                compr_rate: int = 64,
                compr_n_layers: int = None, # only for surgical mistral compressor
                compr_every_n_layer: int = None,
                compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                compr_rms_norm: bool = False, # only for surgical mistral compressor: if true, rms norm applied on h-s
                compr_mlp_hidden_dim: int = 8096,
                compr_use_mlp: bool = True, 
                lora: bool = False, # lora on decoder (and decoder as compr)
                lora_compressor: bool = False, # lora only on the compressor if it exists
                training_form: str = "both",
                lora_r: int = 16,
                lora_r_compressor: int = None,
                load_adapters: bool = True,
                kbtc_training: bool = False,
                optimize_mem_tokens: bool = False,
                different_mem_tokens: bool = False,
                attn_implementation: str = 'flash_attention_2',
                device_map = None,
                **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name # model name of decoder
        self.doc_max_length = doc_max_length # the maximum length of document that can be used by this model (it is used to compute number of mem tokens !)
        self.quantization = quantization # quantization, could be no, int4, int8
        self.sep = sep # boolean type, whether to use sep token
        
        self.compr_model_name = compr_model_name # model name of compressor # null
        self.compr_rate = compr_rate # compression rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        
        self.lora = lora # boolean type, whether to use lora trsining
        self.lora_compressor = lora_compressor
        self.training_form = training_form # training form, could be compressor: training only comprssor; both: training both
        # Or both_separately: training both with separate adapters
        self.lora_r = lora_r # lora_r for lora training, we use 16 throughout the experiment.
        self.lora_r_compressor = lora_r_compressor or lora_r # defaulting to same lora as decoder.
        self.load_adapters = load_adapters # used to load pretrained model: we first load without adapters, and then load them from file.
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        
        self.kbtc_training = kbtc_training
        
        self.device_map = device_map
        
        self.attn_implementation = attn_implementation
        
        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora
            
        
class COCOM(PreTrainedModel):
    config_class = COCOMConfig
    '''
    超级复杂的调用过程，需要了解hf的模型加载机制
    加载 COCOM 模型的标准入口点是其类方法 from_pretrained()。当用户执行 model = COCOM.from_pretrained(llm_model_path) 时，整个模型加载流程便被触发。
        在这个函数内部，会先加载参数config，然后执行初始化函数，最后继续加载模型

    ''' 
    def __init__(self, cfg):
        print("---------------------------- __init__ ----------------------------")
        super().__init__(cfg)
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self.create_decoder(cfg) # 加载base模型，支持量化
        
        self.doc_max_length = cfg.doc_max_length    # 128
        
        print('Base decoder nb parameters', self.decoder.num_parameters())  # 打印解码器参数：7241732096

        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []

        self.chunk_count = [] # 记录文本分块，注意兼容普通版本

        self.compr = None
        print('=============================================================')
        print(cfg)
        print('=============================================================')
        print("cfg.compr_model_name",cfg.compr_model_name)
        # when compr_model_name is not set, then means using a decoder-based compressor, otherwise a bert based compressor
        if cfg.compr_model_name is not None:    # null
            # case bert based compressor
            print('Instantiating compressor ', cfg.compr_model_name)
            self.compr = BertCompressor(cfg.compr_model_name, 
                                        cfg.compr_rate, 
                                        doc_max_length=self.doc_max_length,
                                        decoder_hidden_size=self.decoder.config.hidden_size,
                                        mlp_hidden_dim=cfg.compr_mlp_hidden_dim,
                                        compr_n_layers=cfg.compr_n_layers,
                                        compr_every_n_layer=cfg.compr_every_n_layer,
                                        compr_base_model_name=cfg.compr_base_model_name,
                                        compr_rms_norm=cfg.compr_rms_norm,
                                        use_mlp=cfg.compr_use_mlp,
                                        attn_implementation=cfg.attn_implementation)

        # set lora adaptors on decoder model
        print("cfg.lora",cfg.lora)
        print("cfg.load_adapters",cfg.load_adapters)
        print("self.training_form",self.training_form)

        if cfg.lora:    
            peft_config = self.get_peft_config(lora_r=cfg.lora_r)

            if cfg.load_adapters:  # from_pretrained中设置成false了
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.decoder.set_adapter('decoder_adapter') # active adapter by default # 先不激活
                self.adapter_keys.append('decoder_adapter')
            print("1")

            # Create separate adapters (if not BERT compressor and training_form == 'both_separately')
            if self.training_form == 'both_separately' and self.compr is None:
                if cfg.load_adapters:
                    self.decoder.add_adapter(peft_config, 'encoder_adapter')
                    self.adapter_keys.append('encoder_adapter')
            print("2")
            # 加载一个LLM，两个lora适配器

        # set lora adapters on compressor model:
        if cfg.lora_compressor and self.compr is not None and cfg.load_adapters:    # false
            peft_config = self.get_peft_config(lora_r=cfg.lora_r_compressor)
            self.compr.set_lora(peft_config)
        
        print('Base decoder nb parameters', self.decoder.num_parameters())  # 打印解码器参数：7241732096

        self.decoder_tokenizer = COCOM.create_decoder_tokenizer(cfg)    # 插入mem token

        # resize the tokenizer embedding
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id
        
        # 参数量增加 11*4096*2 新增11个token，维度为4096，2表示嵌入层和lm_head
        print('Base decoder nb parameters', self.decoder.num_parameters())  # 7241822208
        # self.decoder.gradient_checkpointing_enable()
        # if self.compr is not None:
        #     self.compr.gradient_checkpointing_enable()

        # other settings
        self.generation_top_k = 1
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')
        
        self.n_mem_tokens = self.doc_max_length // self.compr_rate # crucial!   # 16倍压缩，128文档转换为8个向量

        print("self.lora:",self.lora)
        print("self.adapter_keys:",self.adapter_keys)

        if self.lora:   # true
            for adapter_key in self.adapter_keys:   # 空 ：[]
                self.decoder.set_adapter(adapter_key)
                print(f'Adapter {adapter_key} trainable parameters: {self.num_parameters(only_trainable=True)}')
                
            #  We need to activate all adapters so that they are both trained...
            self.set_all_adapters()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')
            
        if self.compr is not None:
            print(f'Compressor number of parameters: {self.compr.model.num_parameters(only_trainable=True)}')

        print('Base decoder nb parameters', self.decoder.num_parameters())  # 7241822208
        print('-=-=-==-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-==-=-=-=-=-=-=-=-=-=-=-')

        self.prepare_mem_tokens_optimization()
        # print('Base decoder nb parameters', self.decoder.num_parameters())  # 7241822208
            
    def prepare_mem_tokens_optimization(self):
        if self.config.optimize_mem_tokens: # true
            if self.compr is None:
                # Enforcing gradients for input embeddings (even if lora)
                self.decoder.get_input_embeddings().weight.requires_grad = True
                # Applying a hook zero-ing the gradients except for the mem token:
                def hook(grad):
                    mask = torch.zeros_like(grad)
                    mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                    return grad * mask
                self.decoder.get_input_embeddings().weight.register_hook(hook)
                
    def set_all_adapters(self):
        if len(self.adapter_keys) > 0:
            print('set_adapter:',self.adapter_keys)
            self.decoder.set_adapter(self.adapter_keys)
            
    @staticmethod
    def create_decoder_tokenizer(cfg: COCOMConfig):
        decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name, use_fast=True, padding_side='left')

        # define special tokens
        n_mem_tokens = cfg.doc_max_length // cfg.compr_rate # 128/16 = 8
        if cfg.different_mem_tokens:
            # estimation fo the number of memory tokens needed:
            mem_tokens = ['<MEM' + str(i) + '>' for i in range(n_mem_tokens)]
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': mem_tokens + ['<AE>', '<ENC>', '<SEP>']}) 
            decoder_tokenizer.mem_tokens = mem_tokens
        else:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>', '<AE>', '<ENC>', '<SEP>']})
            decoder_tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens
        
        decoder_tokenizer.mem_token_ids = [decoder_tokenizer.convert_tokens_to_ids(elt) for elt in decoder_tokenizer.mem_tokens]
        decoder_tokenizer.mem_token_ids_pt = torch.LongTensor(decoder_tokenizer.mem_token_ids) # required later on for operations on tensors
        
        decoder_tokenizer.ae_token = '<AE>' # token for autoencoding on decoder side
        decoder_tokenizer.ae_token_id = decoder_tokenizer.convert_tokens_to_ids('<AE>')
        decoder_tokenizer.enc_token = '<ENC>' # token for autoencoding on compressor side
        decoder_tokenizer.sep_token = '<SEP>' # sep token between document
        decoder_tokenizer.sep_token_id = decoder_tokenizer.convert_tokens_to_ids('<SEP>')

        # If kbtc training, we add another one yet
        if cfg.kbtc_training:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            decoder_tokenizer.kbtc_token = '<KBTC>'
            decoder_tokenizer.kbtc_token_id = decoder_tokenizer.convert_tokens_to_ids('<KBTC>')

        # if pad token exists then use pad token, othrwise bos token
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token_id = decoder_tokenizer.bos_token_id

        return decoder_tokenizer

    def get_peft_config(self, lora_r: int) -> LoraConfig:
        """
        Builds the peft config
        """
        return LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=2*lora_r, target_modules='all-linear', lora_dropout=0.1)

    def create_decoder(self, cfg):
        """
        Loads the base decoder.
        """
        if torch.cuda.is_available():
            if cfg.quantization == "no":
                # print(self.config.attn_implementation)
                # print(cfg.device_map)
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name, # model/llm/Mistral-7B-Instruct-v0.2
                    torch_dtype=torch.bfloat16,
                    attn_implementation=self.config.attn_implementation,
                    # low_cpu_mem_usage = True,
                    device_map=cfg.device_map
                    )
            
            elif cfg.quantization == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                    # low_cpu_mem_usage = True,
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    # low_cpu_mem_usage = True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            elif cfg.quantization == "int8":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_compute_dtype='bfloat16',
                    # low_cpu_mem_usage = True,
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    # low_cpu_mem_usage = True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            else:
                raise NotImplementedError()
        else:
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                # low_cpu_mem_usage = True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
            
    def compress(self, enc_input_ids, enc_attention_mask):
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self.compr_decoder(enc_input_ids, enc_attention_mask)

    def replace_emb(self, compressed_embs, dec_input_ids):
        """
        Compression logic (either with decoder or with dedicated compressor)
        """
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)            
        input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        return input_embeds

    def replace_emb1(self, compressed_embs, dec_input_ids):  # 修改版：首位置自寻【实现上是兼容原版的】
        """
        Compression logic (either with decoder or with dedicated compressor)
        """
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k) 

        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)  # 获取嵌入
        num_embs = compressed_embs.size(1)  # 8
        batch_size = inputs_embeds.size(0)  # 通常为 1
        # if self.sep:
        #     slot_len = num_embs + 1 # 9
        # else:
        #     slot_len = num_embs     # 8
        # get first mem_token indices
        # first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1)

        mask = dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]
        all_mem_token_indices = torch.nonzero(mask, as_tuple=False) # 每个元素可以理解为横纵坐标xy
        # print(all_mem_token_indices.shape)  # 2，2
        all_positions = all_mem_token_indices[:, 1].reshape(batch_size, self.generation_top_k)  # 获取每个样本中的序列位置

        
        # for each example in batch, replace them with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = all_positions[i][j-indices[i]].item()
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size(), \
                    f"{inputs_embeds[i, start_idx:start_idx + num_embs, :].size()} VS {compressed_embs[j].size()}"  # 4096
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds



    def compr_decoder(self, input_ids, attention_mask):
        """
        Compression using the decoder
        利用解码器（self.decoder）对输入进行压缩处理，同时只保留与记忆标记（mem_tokens）相关的嵌入向量。
        """
        assert input_ids.size() == attention_mask.size(), f"{input_ids.size()} vs {attention_mask.size()}"
        
        # get_gpu_memory_info_torch()   # 13G
        # Switch adapter if we are training two different ones:
        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')

        # get_gpu_memory_info_torch()
        # print(input_ids.shape)  # 6,1035【batch=6，1035=1024+8+3】
        
        self.decoder.eval()
        with torch.no_grad():  # 禁用梯度   # 只是一次prefill，那为什么这么慢呢？
            emb = self.decoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                        #    use_cache=False,
                           output_hidden_states=True).hidden_states[-1] # 解码器最后一层隐藏向量(batch_size, sequence_length, hidden_size)
            # emb = self.decoder(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     use_cache=False,
            #     output_hidden_states=True,          # 关键：只返回 last_hidden_state
            # )
            # emb = out.hidden_states[-1]
            # attention_avg = torch.stack(out.hidden_states[1:]).mean(dim=0)
        
        # 0 7 15 -1
        # print(emb.shape)
        # get_gpu_memory_info_torch()

        mask = torch.isin(input_ids, self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device))
        return emb[mask].reshape(emb.size(0), -1, emb.size(-1)) #, attention_avg[mask].reshape(emb.size(0), -1, emb.size(-1))
    
    def prepare_encoder_inputs_to_decoder(self, texts, max_length, q_texts=None):
        # texts文档块列表，max_length：文档块最大长度
        if q_texts is not None:
            texts_to_encode = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + '\nQuery:\n' + query + 'Document:\n' + text + self.decoder_tokenizer.eos_token 
                               for text, query in zip(texts, q_texts)]
            inp_enc = self.decoder_tokenizer(texts_to_encode, return_tensors='pt', padding='max_length', max_length=max_length + 8, truncation=True, add_special_tokens=False)
        else:
            inp_enc = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + text + self.decoder_tokenizer.eos_token for text in texts]
            # 分词
            inp_enc = self.decoder_tokenizer(inp_enc, return_tensors='pt', padding="max_length", max_length=max_length+3, truncation=True, add_special_tokens=False)
        
        # print("inp_enc['input_ids']:",inp_enc['input_ids'].shape)   # 12 131(取决于max_length)
        num_mem_tokens = self.doc_max_length // self.compr_rate
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)

        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'], 
                                                                                        inp_enc['attention_mask'], 
                                                                                        num_mem_tokens, 
                                                                                        tokenizer=self.decoder_tokenizer)
        
        return inp_enc
    
    def prepare_encoder_inputs(self, texts: List[str], max_length: int, q_texts: List[str] = None):
        """
        Create the inputs to the encoder, for compression.
        """
        if q_texts is not None:
            assert len(texts) == len(q_texts), f"{len(texts)} == {len(q_texts)}"

        if self.compr is None:  # Case where the encoder is the decoder with adapter:
            return self.prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        else:   # Case where the encoder is a separate network:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def replace_embeddings(self, compressed_embs, dec_input_ids, indices):
        """
        Replace memory tokens in the decoder input to with the compressed embeddings
        """
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        if self.sep:
            slot_len = num_embs + 1
        else:
            slot_len = num_embs
        # get first mem_token indices
        first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1)
        batch_size = inputs_embeds.size(0)
        # for each example in batch, replace them with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j-indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size(), \
                    f"{inputs_embeds[i, start_idx:start_idx + num_embs, :].size()} VS {compressed_embs[j].size()}"
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds

    def forward(self,
                enc_input_ids: torch.LongTensor = None,
                enc_attention_mask: torch.LongTensor = None,
                dec_input_ids: torch.LongTensor = None,
                dec_attention_mask: torch.LongTensor = None,
                labels: torch.LongTensor = None):
        """
        enc_input_ids: stores the contexts, should be flattened from all queries before input, can be of shape:
            - (batch_size*generation_top_k, enc_token_length)
            - (batch_size, generation_top_k, enc_token_length)
        enc_attention_mask: attention mask of enc_input_ids, same shape as enc_input_ids
        dec_input_ids: stores the prompts (including mem tokens), dimention (batch_size, dec_token_length)
        dec_attention_mask: attention mask of dec_input_ids
        """ 
        assert enc_input_ids.size() == enc_attention_mask.size(), f"{enc_input_ids.size()} vs {enc_attention_mask.size()}"
        
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
        
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \
            f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
            
        # Perform compression with gradient tracking
        compressed_embs = self.compress(enc_input_ids, enc_attention_mask)
        inputs_embeds = self.replace_emb(compressed_embs, dec_input_ids)

        # if training_form is compressor, then detach the inputs_embeds, to make gradient not count in decoder
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds  = inputs_embeds.detach()

        # decoding
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_attention_mask, labels=labels)

        # At end of forward, we need to activate all adapters so that they are both trained...
        self.set_all_adapters()

        return {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}
    
    def similarity_last_vs_rest(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return : list[float]，最后一位与保留文档的余弦相似度，保留2位小数
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        # 2. 每个文档的 8 个向量取平均
        kept_avg = tensors.mean(dim=1)                   # [new_num, 4096]
        last_avg = tensors[-1].mean(dim=0)             # [4096]

        # 3. L2 归一化
        kept_norm = torch.nn.functional.normalize(kept_avg, dim=1)
        last_norm = torch.nn.functional.normalize(last_avg, dim=0)

        # 4. 计算点积（余弦相似度）
        sims = torch.matmul(kept_norm, last_norm)     # [new_num]
        retult = [round(float(s), 2) for s in sims]
        # print(retult)

        return retult
    
    def similarity_last_vs_rest_concat(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return : list[float]，最后一位与保留文档的余弦相似度，保留2位小数
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        # 2. 每个文档的 8 个向量拼接
        kept_concat = tensors.view(tensors.size(0), -1)      # [new_num, 32768]
        last_concat = tensor[-1].view(-1)            # [4096]

        # 3. L2 归一化
        kept_norm = torch.nn.functional.normalize(kept_concat, dim=1)
        last_norm = torch.nn.functional.normalize(last_concat, dim=0)

        # 4. 计算点积（余弦相似度）
        sims = torch.matmul(kept_norm, last_norm)     # [new_num]
        retult = [round(float(s), 2) for s in sims]
        # print(retult)

        return retult

    def similarity_last_vs_rest_max(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return : list[float]，最后一位与保留文档的余弦相似度，保留2位小数
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        # 2. 每个文档的 8 个向量取平均
        kept = tensors       
        last = tensors[-1]           

        # 3. L2 归一化
        kept_norm = torch.nn.functional.normalize(kept, dim=-1)
        last_norm = torch.nn.functional.normalize(last, dim=-1)

        # 4. 计算点积（余弦相似度）
        sims = torch.matmul(kept_norm, last_norm.T) 
        max_sims, _ = sims.view(sims.size(0), -1).max(dim=1)
        retult = [round(float(s), 2) for s in max_sims]
        # print(retult)

        return retult

    def generate(self, model_input, return_doc_embeddings: bool = False, **kwargs):    # return_doc_embeddings：False

        enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask'], model_input['dec_input_ids'], model_input['dec_attention_mask']
        
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        # 三维
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
            
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        # assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \  # 为了检索临时取消
        #     f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
        
        # print("enc_input_ids.shape:",enc_input_ids.shape)   # num_of_doc,139(128+11)
        # 使用解码器计算获取软向量【就是这里为什么会占用如此之大的显存呢，并且似乎在后续推理时持续占用：没有关闭梯度计算】
        # print("enc_input_ids:",enc_input_ids.shape)
        st = time.time()
        compressed_embs = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        used_time = round(time.time()-st, 2)
        print(f"embed time:{used_time}")
        # print("compressed_embs:",compressed_embs.shape) # [num_of_doc, 8, 4096]

        # 返回嵌入向量，其中<mem>已经替换成soft向量
        # print(compressed_embs.shape)
        # print(dec_input_ids.shape)
        inputs_embeds = self.replace_emb1(compressed_embs, dec_input_ids.to('cuda'))
        # print("inputs_embeds:",inputs_embeds.shape)

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 
        st = time.time()
        # print(inputs_embeds.shape)
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=dec_attention_mask.to("cuda"),
            # do_sample=False,
            # top_p=None,
            # max_new_tokens=max_new_tokens
            **kwargs
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        used_time = round(time.time()-st, 2)
        print(f"decode time:{used_time}")

        if return_doc_embeddings:   # 用的话改成filtered_compressed_embs
            # Compressed_embds is of shape (batch_size*top_k, n_mem_tokens, hidden_dim)
            # We reshape to batch_size, top_k, n_mem_tokens, hidden_dim

            assert batch_size is not None
            assert top_k is not None
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        else:
            return decoded


    
    def generate_from_text_w_Retrieval(self, questions: List[str], documents: List[List[str]], mode: int = 0, **kwargs) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        只压缩对话中助手回复的内容
        plus 检索

        """

        # self.generation_top_k = (len(documents[0]) // 2 ) + 1

        # 问题与文档对应
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # 划分用户提问和模型回复【第一个是系统提示】
        comp_list, prot_list = self.split_by_odd_even(documents)
        


        # 
        # assert self.generation_top_k == len(comp_list[0])

        # 全部都要压缩，用于计算相似度 检索
        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        flat_documents = flat_documents + questions 
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量

        # 分词 token编码
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=128)    # 128 压缩文档长度限制 可以修改成512之类的

        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # 压缩
        compressed_embs = self.compress(model_input['enc_input_ids'], model_input['enc_attention_mask'])

        # 过滤
        if mode == 0:
            result = self.similarity_last_vs_rest(compressed_embs)
        elif mode == 1:
            result = self.similarity_last_vs_rest_concat(compressed_embs)  # similarity_last_vs_rest_concat similarity_last_vs_rest_max
        elif mode == 2:
            result = self.similarity_last_vs_rest_max(compressed_embs)
        # print(retult)
        print('--------------------------')
        print(result)
        print('--------------------------')

        final_prot_list = [[]]
        final_compressed_embs = compressed_embs[0:1]
        # 极简规则：大于系统提示则压缩保留，否则丢弃
        if len(result) > 2:
            t = result[0]
            for i in range(1,len(result)-1,2):
                if result[i] >= t or result[i+1] >= t:
                    final_prot_list[0].append(prot_list[0][i//2])
                    final_compressed_embs = torch.cat([final_compressed_embs, compressed_embs[i+1:i+2]], dim=0)
        print(type(final_compressed_embs))

        # assert len(final_compressed_embs)-1 == len(final_prot_list[0]), "match 1156"
        print(final_compressed_embs.shape)
        print(len(final_prot_list[0]))
        print('---------------------------------------')

        self.generation_top_k = final_compressed_embs.shape[0]

        # Creating decoder inputs 拼接prompt
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc(query=q,doc=final_prot_list[i]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        # model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # 替换占位符为压缩向量
        inputs_embeds = self.replace_emb1(final_compressed_embs, inp_dec['input_ids'].to(device))
        # print("inputs_embeds:",inputs_embeds.shape)

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 
        st = time.time()
        # print(inputs_embeds.shape)
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=inp_dec['attention_mask'].to(device),
            # do_sample=False,
            # top_p=None,
            # max_new_tokens=max_new_tokens
            **kwargs,
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        used_time = round(time.time()-st, 2)
        print(f"decode time:{used_time}")


        # Generation
        # return self.generate(model_input, max_new_tokens=max_new_tokens, mode=mode), instr
        
        return decoded ,instr


    def generate_from_text_w_Retrieval_1(self, questions: List[str], documents: List[List[str]], mode: int = 0, **kwargs) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        只压缩对话中助手回复的内容
        plus 检索
        第一版：完全丢弃/压缩/不压缩
        """

        # self.generation_top_k = (len(documents[0]) // 2 ) + 1

        # 问题与文档对应
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # 划分用户提问和模型回复【第一个是系统提示】
        comp_list, prot_list = self.split_by_odd_even(documents)
        


        # 
        # assert self.generation_top_k == len(comp_list[0])

        # 全部都要压缩，用于计算相似度 检索
        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        flat_documents = flat_documents + questions 
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量

        # 分词 token编码
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=128)    # 128 压缩文档长度限制 可以修改成512之类的

        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # 压缩
        compressed_embs = self.compress(model_input['enc_input_ids'], model_input['enc_attention_mask'])

        # 过滤
        if mode == 0:
            result = self.similarity_last_vs_rest(compressed_embs)
        elif mode == 1:
            result = self.similarity_last_vs_rest_concat(compressed_embs)  # similarity_last_vs_rest_concat similarity_last_vs_rest_max
        elif mode == 2:
            result = self.similarity_last_vs_rest_max(compressed_embs)
        # print(retult)
        print('--------------------------')
        print(result)
        print('--------------------------')

        ind = [[1]]
        final_compressed_embs = compressed_embs[0:1]

        # 0丢弃/1压缩/3不压缩
        if len(result) > 2:
            t1, t2 = 0.93, 0.95 # mode=1
            for i in range(1,len(result)-1,2):
                if result[i] < t1 and result[i+1] < t1:    # 完全丢弃
                    ind[0].append(0)
                    ind[0].append(0)
                    # pass
                elif result[i] > t2 and result[i+1] > t2: # 不压缩
                    ind[0].append(3)
                    ind[0].append(3)
                else:   # 压缩
                    ind[0].append(3)
                    ind[0].append(1)
                    final_compressed_embs = torch.cat([final_compressed_embs, compressed_embs[i+1:i+2]], dim=0)

        print(type(final_compressed_embs))

        # assert len(final_compressed_embs)-1 == len(final_prot_list[0]), "match 1156"
        print(final_compressed_embs.shape)
        # print(len(final_prot_list[0]))
        print('---------------------------------------')

        self.generation_top_k = final_compressed_embs.shape[0]

        # Creating decoder inputs 拼接prompt
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc_1(query=q,doc=documents[i],ind=ind[i]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        # model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # 替换占位符为压缩向量
        inputs_embeds = self.replace_emb1(final_compressed_embs, inp_dec['input_ids'].to(device))
        # print("inputs_embeds:",inputs_embeds.shape)

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 
        st = time.time()
        # print(inputs_embeds.shape)
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=inp_dec['attention_mask'].to(device),
            # do_sample=False,
            # top_p=None,
            # max_new_tokens=max_new_tokens
            **kwargs,
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        used_time = round(time.time()-st, 2)
        print(f"decode time:{used_time}")


        # Generation
        # return self.generate(model_input, max_new_tokens=max_new_tokens, mode=mode), instr
        
        return decoded ,instr


    def generate_from_text_w_Retrieval_2(self, questions: List[str], documents: List[List[str]], mode: int = 0, **kwargs) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        只压缩对话中助手回复的内容
        plus 检索
        第2版：丢弃模型回复部分/压缩/不压缩【压缩部分改成分块压缩】
        1024长度
        """

        # 问题与文档对应
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # 划分用户提问和模型回复【第一个是系统提示】
        comp_list, prot_list = self.split_by_odd_even(documents)
        # 

        # 全部都要压缩，用于计算相似度 检索
        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        flat_documents = flat_documents + questions 
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量

        # 分词 token编码
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=512)    # 128 压缩文档长度限制 可以修改成512之类的

        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token


        # 压缩
        compressed_embs = self.compress(model_input['enc_input_ids'], model_input['enc_attention_mask'])


        # 过滤
        if mode == 0:   # 利用最近对话来一起检索
            result = self.similarity_last_vs_rest(compressed_embs)  # attention_avg compressed_embs
        elif mode == 1:
            result = self.similarity_last_vs_rest_concat(compressed_embs)  # similarity_last_vs_rest_concat similarity_last_vs_rest_max
        elif mode == 2:
            result = self.similarity_last_vs_rest_max(compressed_embs)
        # print(retult)
        print('--------------------------')
        print(result)
        print('--------------------------')

        ind = [[1,3,3]]
        final_compressed_embs = compressed_embs[0:1]

        # 0丢弃/1压缩/3不压缩
        if len(result) > 4:
            t1, t2 = 0.5, 0.8 # mode=1
            # 可以看到，指令完全不丢弃，那么为什么还要比较指令的相关性呢=》丢弃模型回复，形成的上下文大部分是指令，效果并不是很好
            # 最理想的情况还是指令与模型回复都丢弃；但是一些全局性的指令丢了就错了
            for i in range(3,len(result)-1,2):
                if result[i+1] < t1:    # 完全丢弃 result[i] < t1 and 
                    ind[0].append(0)  # 指令不丢 [第一个不丢的情况下其他可以丢]
                    ind[0].append(0)
                elif result[i+1] > t2: # 不压缩 result[i] > t2 and
                    ind[0].append(3)
                    ind[0].append(3)
                else:   # 压缩
                    ind[0].append(3)
                    ind[0].append(1)
                    final_compressed_embs = torch.cat([final_compressed_embs, compressed_embs[i+1:i+2]], dim=0)

        # print(type(final_compressed_embs))

        # assert len(final_compressed_embs)-1 == len(final_prot_list[0]), "match 1156"
        print(final_compressed_embs.shape)
        # print(len(final_prot_list[0]))
        print('---------------------------------------')

        self.generation_top_k = final_compressed_embs.shape[0]

        # Creating decoder inputs 拼接prompt
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc_1(query=q,doc=documents[i],ind=ind[i][:len(documents[i])]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        # model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # 替换占位符为压缩向量
        inputs_embeds = self.replace_emb1(final_compressed_embs, inp_dec['input_ids'].to(device))
        # print("inputs_embeds:",inputs_embeds.shape)

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 
        st = time.time()
        # print(inputs_embeds.shape)
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=inp_dec['attention_mask'].to(device),
            # do_sample=False,
            # top_p=None,
            # max_new_tokens=max_new_tokens
            **kwargs,
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        used_time = round(time.time()-st, 2)
        print(f"decode time:{used_time}")

        return decoded, instr


    def generate_from_full_text(self, questions: List[str], documents: List[List[str]], max_new_tokens: int = 1024) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        max_new_tokens : 128
        """
        history = documents[0]
        current_question = questions[0]
        # 将历史对话和当前问题合并成一个输入序列
        docs = " ".join(history).strip() # prompt没有对齐
        
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        # prompt_user = f"\n\nBackground:\n{docs}\n\nQuestion:{question}"
        prompt_user = f"{prompt_system}\n\nBackground:\n{docs}\n\nQuestion:{current_question}"
        
        # Prepare the messages with system and user roles
        # messages = [
        #     {"role": "system", "content": prompt_system},
        #     {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
        # ]

        # 构造单一用户角色的消息
        messages = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]
        # 分词处理（根据具体模型的分词器进行调整）
        prompt = [self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inputs = self.decoder_tokenizer(prompt, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        # print(inputs["input_ids"].shape)
        # inputs = self.decoder_tokenizer(prompt, return_tensors="pt").to('cuda')
        inputs_embeds = self.decoder.get_input_embeddings()(inputs["input_ids"].to('cuda'))
        # if 'decoder_adapter' in self.adapter_keys:
        #     self.decoder.set_adapter('decoder_adapter') 

        # # 生成回复
        # outputs = self.decoder.generate(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"],
        #     max_length=512,  # 设定生成的最大长度
        #     temperature=0.7,  # 控制生成的随机性
        #     top_k=50,  # 只从 top_k 个 token 中采样
        #     top_p=None,  # 只从概率和超过 top_p 的 token 中采样
        #     pad_token_id=self.decoder_tokenizer.eos_token_id,
        #     do_sample=False  # 随机采样生成
        # )

        # 解码生成的回复
        # response = self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # # 移除输入部分，只保留生成的回复
        # response = response[len(prompt):].strip()

        # Switch adapter if we are training two different ones:
        
        # if 'decoder_adapter' in self.adapter_keys:
        #     self.decoder.set_adapter('decoder_adapter') 

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=inputs["attention_mask"].to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
            )
        
        response = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return response,prompt


    def embeds(self, model_input, max_new_tokens=128, return_doc_embeddings: bool = True):    # return_doc_embeddings：False

        enc_input_ids, enc_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask']
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        # batch大小，或者说是文档数目
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
            
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        # assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \
        #     f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
        
        # print("enc_input_ids.shape:",enc_input_ids.shape)   # num_of_doc,139(128+11)
        # 使用解码器计算获取软向量【就是这里为什么会占用如此之大的显存呢，并且似乎在后续推理时持续占用：没有关闭梯度计算】
        # print("enc_input_ids:",enc_input_ids.shape)
        compressed_embs = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        # print("compressed_embs:",compressed_embs.shape) # [num_of_doc, 8, 4096]

        # assert 1==2

        return compressed_embs


    def get_all_adapters_state_dict(self):
        """
        Return the state dicts of the adapters
        Used for saving so we go to cpu automatically
        """
        return {key: {k:v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()} for key in self.adapter_keys}

    def load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: dict) -> None:
        """
        Creates an adapter from the state dict (used to load from pretrained)
        """
        # assert adapter_name not in self.adapter_keys, f'Adapter {adapter_name} already exists'
        print(f'loading adapter {adapter_name}')
        self.decoder.load_adapter(peft_config=peft_config, adapter_name=adapter_name, adapter_state_dict=adapter_state_dict)
        self.adapter_keys.append(adapter_name)
        
    def get_decoder_first_and_last_layer_state_dict(self) -> dict:
        """
        Just getting the first and last layers: the only ones which change when adding tokens
        Used to save the model so we automatically move to cpu.
        """
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
                
        # assert len(out) == 2, len(out) # We should get both the embedding layer and the head layer.
        return out

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save only the LoRA adapters and their configurations.
        """
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory) 

            # Save the LoRA adapter weights
            torch.save(self.get_all_adapters_state_dict(), os.path.join(save_directory, "adapters.pth"))
            
            # Save the first and last layers of decoder (because of diffs with tokens !)
            torch.save(self.get_decoder_first_and_last_layer_state_dict(), os.path.join(save_directory, "decoder_first_last_layers.pth"))
            
            # Save the bert compressor if it exists
            if self.compr_model_name is not None:
                self.compr.save_pretrained(os.path.join(save_directory, 'compressor'))

            # Save the configuration
            self.config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        """
        print("---------------------------- from_pretrained ----------------------------")
        # Load the configuration
        config = COCOMConfig.from_pretrained(pretrained_model_name_or_path)
        # print('111111111111111111111111111111111111111111111111111111111111111111111')
        config.attn_implementation = kwargs.get('attn_implementation', config.attn_implementation)  
        # print('222222222222222222222222222222222222222222222222222222222222222222222')
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        # print('3333333333333333333333333333333333333333333333333333333333333333333333')
        if config.lora:
            # print('444444444444444444444444444444444444444444444444444444444444444444444444444444')

            # We need to delay the construction of the adapters (otherwise peft complains)
            config.load_adapters = False    # 延迟适配器加载

            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize the model
            model = cls(config) # 转到初始化函数
            print('model nb parameters', model.num_parameters()) 
            # print('55555555555555555555555555555555555555555555555555555555555555555555555555555555555')

            # Loading first and last layers (they might have changed due to extra tokens)
            try:
                # If loading from Hugging Face Hub
                first_and_last_layers_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="decoder_first_last_layers.pth"
                )
            except Exception as e:
                # If loading from a local directory
                first_and_last_layers_path = os.path.join(pretrained_model_name_or_path, "decoder_first_last_layers.pth")
            print("替换首尾层模型")

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(first_and_last_layers_path, map_location=map_location, weights_only=True)
                for key in first_and_last_decoder_state_dict:
                    assert key in model.decoder.state_dict()
                    model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)
            else:
                print('FIRST AND LAST LAYER NOT FOUND (ok for some old models):', first_and_last_layers_path)
            print('1. model nb parameters:', model.num_parameters())    # 7241822208
            peft_config = model.get_peft_config(lora_r=config.lora_r)
            
            # Load the LoRA adapters (if the file exists)
            try:
                # If loading from Hugging Face Hub
                adapters_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="adapters.pth"
                )
                # print("加载适应模型1")
            except Exception as e:
                # If loading from a local directory
                adapters_path = os.path.join(pretrained_model_name_or_path, "adapters.pth")
            print("加载适应模型")
                
            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
            
                for key, val in adapters_state_dict.items():
                    model.load_adapter_from_state_dict(peft_config=peft_config, adapter_name=key, adapter_state_dict=val)
                print("load adapter")
            else:
                warnings.warn(f'I see lora on that PISCO model, but {adapters_path} does not exist, it may be normal \
                        for recent versions of transformers, be aware.')
            print('2. model nb parameters', model.num_parameters())     # 7325708288
            # If there is a compressor, it's been built: we just need to load the state dict or the adapters:
            if config.compr_model_name is not None: # null
                model.compr.load_pretrained(os.path.join(pretrained_model_name_or_path, 'compressor'), 
                                            lora=config.lora_compressor, 
                                            peft_config=model.get_peft_config(lora_r=config.lora_r_compressor))
            print('3. model nb parameters', model.num_parameters()) 
            model.set_all_adapters()

            model.config.load_adapters = True
            print('model nb parameters', model.num_parameters()) 
            # print('Base decoder nb parameters', self.decoder.num_parameters()) 

            return model

        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
    def generate_from_text(self, questions: List[str], documents: List[List[str]], max_new_tokens: int = 128) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        """
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。
        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        
        model_input = {}
        
        # Creating encoder inputs:
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=512)    # 128 压缩文档长度限制 128
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # Creating decoder inputs
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens(query=q) for q in questions]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制  2048
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self.generate(model_input, max_new_tokens=max_new_tokens)    

    def blend_prompt_and_memory_tokens_with_doc_1(self, query: str, doc: List[str], ind: List[int]):
        """
        Takes care of blending the prompt with the memory tokens:
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        (Used for the HUB version)
        模型回复使用占位符，用户指令使用原始文本

        为检索设计：传入所有文档和对应的状态
        doc:包含系统指令和所有用户提问 模型回复

            0   丢弃
            1   压缩
            2   分块压缩
            3   完全保留
        """        
        assert len(doc) == len(ind), 'match_1588'
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        

        docs = ""

        for i in range(1,len(doc)):
            if ind[i] == 0:
                pass
            elif ind[i] == 1:
                docs += mem_tokens_str
            elif ind[i] == 2:   # 不太好实现，需要二次压缩
                docs += mem_tokens_str
            elif ind[i] == 3:
                docs += doc[i]
        
        # prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        # prompt_user = f"\n\nBackground:\n{docs}\n\nQuestion:{question}"
        
        prompt_user = f"{mem_tokens_str}\n\nBackground:\n{docs}\n\nQuestion:{query}"
        
        # Prepare the messages with system and user roles
        # messages = [
        #     {"role": "system", "content": prompt_system},
        #     {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
        # ]

        # 构造单一用户角色的消息
        messages = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        # Attempt to apply the system role and catch if it's not supported
        try:
            # 将聊天消息格式化为模型可以理解的输入文本。这个函数通常会根据特定的模板将用户和助手的消息组合成一个统一的输入字符串。
            prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                # Apply template again without system role
                prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        # print("---------------------prompt-------------------------------")
        # print(prompt)

        return prompt


    def blend_prompt_and_memory_tokens_with_doc(self, query: str, doc: List[str]):
        """
        Takes care of blending the prompt with the memory tokens:
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        (Used for the HUB version)
        模型回复使用占位符，用户指令使用原始文本
        """        
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        
        # proper names for "eval" call, don't remove these lines
        # docs = mem_tokens_str * self.generation_top_k
        # docs = mem_tokens_str.join(doc)
        docs = ""
        for i in doc:
            # print(i)
            docs += i + mem_tokens_str
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        question = query
        
        # prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        # prompt_user = f"\n\nBackground:\n{docs}\n\nQuestion:{question}"
        
        prompt_user = f"{mem_tokens_str}\n\nBackground:\n{docs}\n\nQuestion:{question}"
        
        # Prepare the messages with system and user roles
        # messages = [
        #     {"role": "system", "content": prompt_system},
        #     {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
        # ]

        # 构造单一用户角色的消息
        messages = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        # Attempt to apply the system role and catch if it's not supported
        try:
            # 将聊天消息格式化为模型可以理解的输入文本。这个函数通常会根据特定的模板将用户和助手的消息组合成一个统一的输入字符串。
            prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                # Apply template again without system role
                prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        # print("---------------------prompt-------------------------------")
        # print(prompt)

        return prompt

    def get_prompt(self, messages, system_prompt=None):
        seps = [" ", "</s>"]
        roles=("[INST]", "[/INST]")
        if system_prompt:
            ret = f"[INST] {system_prompt}\n"
        else:
            ret = "[INST] "

        for i, message in enumerate(messages):
            tag = roles[i % 2]
            if i == 0:  # 前面已经有inst了
                ret += message['content'] + " "
            else:
                ret += tag + " " + message['content'] + seps[i % 2]

        ret += roles[1]
    # print('#################################################################')
        # print(ret)
    # print('-----------------------------------------------------------------')
        return ret


    def blend_prompt_and_memory_tokens_with_doc_inst(self, query: str, doc: List[str]):
        """
        Takes care of blending the prompt with the memory tokens:
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        (Used for the HUB version)
        模型回复使用占位符，用户指令使用原始文本
        转为多轮对话设计，包含[inst]和[\inst]
        """        
        # 单个128回复的占位符：[mem]*8+<sep>
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token

        messages = [
            {"role": "system", "content": mem_tokens_str * self.chunk_count[0]},
        ]

        # print('------------------------doc-----------------------------')
        # print(doc)
        # print('------------------------doc-----------------------------')
        for i, use_query in enumerate(doc):
            message = [
                {"role": "user", "content": use_query},
                {"role": "assistant", "content": mem_tokens_str * self.chunk_count[i+1]}
            ]
            messages += message

        # 构造单一用户角色的消息
        messages += [
            {"role": "user", "content": query}
        ]
        # print(messages)

        # Attempt to apply the system role and catch if it's not supported
        # try:
        #     # 将聊天消息格式化为模型可以理解的输入文本。这个函数通常会根据特定的模板将用户和助手的消息组合成一个统一的输入字符串。
        #     prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       
        # except TemplateError as e:
        #     # Catch the error related to system role and handle it (e.g. gemma)
        #     # if "System role not supported" in str(e):   # 确实不支持system
        #         # Remove system role and proceed with only the user role
        #     full_messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
        #     if len(messages)>2:
        #             full_messages += messages[2:]
        #     prompt = self.decoder_tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
        prompt = self.get_prompt(messages[1:],messages[0]['content'])
        return prompt


    def split_by_odd_even(self, elements):
        even_list = []  # 存储偶索引的元素
        odd_list = []   # 存储奇索引的元素

        for sublist in elements:
            even_sub = []
            odd_sub = []
            for index, item in enumerate(sublist):
                if index % 2 == 0:
                    even_sub.append(item)
                else:
                    odd_sub.append(item)
            even_list.append(even_sub)
            odd_list.append(odd_sub)

        return even_list, odd_list  # MTEval使用，包含系统指令


    def generate_from_text1(self, questions: List[str], documents: List[List[str]], **kwargs) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        只压缩对话中助手回复的内容
        """
        self.generation_top_k = (len(documents[0]) // 2 ) + 1
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # for i in documents[0]:
        #     print(i)
        #     print("------------0--------------")

        comp_list, prot_list = self.split_by_odd_even(documents)
        
        # for i in comp_list:
        #     print(i)
        #     print("-------------1-------------")
        
        # for i in prot_list:
        #     print(i)
        #     print("-------------2-------------")
        assert self.generation_top_k == len(comp_list[0])

        flat_documents = sum(comp_list, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=1024)    # 128 压缩文档长度限制 可以修改成512之类的
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # Creating decoder inputs
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc(query=q,doc=prot_list[i]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self.generate(model_input, **kwargs),instr


    def generate_from_text_R(self, questions: List[str], documents: List[List[str]], max_new_tokens: int = 128, mode: int = 0) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        只压缩对话中助手回复的内容
        plus 检索

        # 用于测试，已舍弃
        """

        self.generation_top_k = (len(documents[0]) // 2 ) + 1
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # for i in documents[0]:
        #     print(i)
        #     print("------------0--------------")

        comp_list, prot_list = self.split_by_odd_even(documents)
        
        # for i in comp_list:
        #     print(i)
        #     print("-------------1-------------")
        
        # for i in prot_list:
        #     print(i)
        #     print("-------------2-------------")
        assert self.generation_top_k == len(comp_list[0])

        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        flat_documents = flat_documents + questions # 全部都要压缩
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=128)    # 128 压缩文档长度限制 可以修改成512之类的

        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # Creating decoder inputs
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc(query=q,doc=prot_list[i]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self.generate(model_input, max_new_tokens=max_new_tokens, mode=mode), instr
        

    def split_texts(self, texts: List[str],chunk_size: int = 128) -> Tuple[List[str], List[int]]:
        """
        将字符串列表递归分块，保证每个块不超过 chunk_size 个字符；
        返回：
            1. 所有分块拼成的新字符串列表
            2. 原列表中每个字符串对应的切块数量列表

        注意延时
        """
        # 方法一：用 LangChain + HuggingFace tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.decoder_tokenizer, # mistral分词器+11个特殊向量
            chunk_size=chunk_size,
            chunk_overlap=0,         # 不重叠
            add_start_index=False,   # 不保留元信息
            separators=["\n\n", "\n", ".", " ", ""]
        )
        new_chunks: List[str] = []
        chunk_counts: List[int] = []

        for text in texts:
            docs = splitter.create_documents([text])
            chunks = [doc.page_content for doc in docs]
            new_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        return new_chunks, chunk_counts


    def generate_from_chunk_text(self, questions: List[str], documents: List[List[str]], max_new_tokens: int = 1024, system_prompt: bool=True) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)(包含系统提示)
        只压缩对话中助手回复的内容  分块：128
        """
        # self.generation_top_k = len(documents[0]) // 2 
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。

        # for i in documents[0]:
        #     print(i)
        #     print("------------0--------------")

        if system_prompt:
            comp_list, prot_list  = self.split_by_odd_even(documents)   # 偶数，奇数
        else:
            prot_list, comp_list = self.split_by_odd_even(documents)    # 没有系统指令，用户/模型 交替

        comp_chunk_list, self.chunk_count = self.split_texts(comp_list[0])
        self.generation_top_k = len(comp_chunk_list)

        # print('--------------------------------------------------')
        # print(self.chunk_count)

        # assert self.generation_top_k == len(comp_list[0])
        # for i in comp_list:
        #     print(i)
        #     print("-------------1-------------")
        
        # for i in prot_list:
        #     print(i)
        #     print("-------------2-------------")


        # flat_documents = sum(comp_list, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        
        model_input = {}
        
        # Creating encoder inputs: 将文档压缩成8个向量
        input_encoder = self.prepare_encoder_inputs(comp_chunk_list, max_length=128)    # 128 压缩文档长度限制 可以修改成512之类的
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # Creating decoder inputs
        # 返回LLM可以理解的文本（prompt）：使用<mem>占用文档位置，使用<sep>隔开不同文档（可以设置多文档topk）
        instr = [self.blend_prompt_and_memory_tokens_with_doc_inst(query=q,doc=prot_list[i]) for i,q in enumerate(questions)]
        # 分词
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=8192)    # 查询长度限制 2048
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self.generate(model_input, max_new_tokens=max_new_tokens),instr

       
    def get_embeds(self, documents: List[List[str]]) -> List[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: List of string
        documents: List of List of strings (they should all be of equal length: the nb of doc for each question)
        二维文档
        """
        self.generation_top_k = len(documents[0])
        # assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])    # 每个问题对应的文档数量相同。
        flat_documents = sum(documents, []) # 展平文档列表，输出一维列表：sum(iterable, start)，其中 start 是累加的起始值，默认为 0。在这里，start 被设置为 []，表示从一个空列表开始累加。
        # flat_documents = documents
        
        model_input = {}
        
        # Creating encoder inputs:
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=128)    # 128 压缩文档长度限制
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        # print("generate_from_text   model_input['enc_input_ids']:",model_input['enc_input_ids'].shape)  # 139 = 128 + 3间隔符 + 8个软向量token

        # Generation
        return self.embeds(model_input)    
  

    def generate_from_compressed_documents_and_questions(self, questions: List[str], compressed_documents: torch.Tensor, max_new_tokens: int = 128) -> List[str]:
        """
        Generates answers from compressed documents
        questions: List of string
        compressed_documents: torch tensor, its first dimension should be a multiple of len(questions)
        """
        self.generation_top_k = compressed_documents.size(0) // len(questions)  # 1
        assert compressed_documents.size(0) % self.generation_top_k == 0, f"{compressed_documents.size(0)} {self.generation_top_k}"
        
        # Creating decoder inputs
        instr = [self.blend_prompt_and_memory_tokens(query=q) for q in questions]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=2048)
        device = self.decoder.device
        dec_input_ids, dec_attention_mask = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)

        # Creating input decoder embeddings from prompt + compressed documents
        inputs_embeds = self.replace_emb(compressed_documents, dec_input_ids)
        
        # Activating decoder generator:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
            
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens
            )
        
        # de-tokenizing
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
    def compress_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Compress a List of documents
        """
        input_encoder = self.prepare_encoder_inputs(documents, max_length=1024) # 128
        enc_input_ids = input_encoder['input_ids'].to(self.decoder.device)
        attention_mask = input_encoder['attention_mask'].to(self.decoder.device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)
    
    def blend_prompt_and_memory_tokens(self, query: str):
        """
        Takes care of blending the prompt with the memory tokens:
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        (Used for the HUB version)
        """        
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        # print(mem_tokens_str)   # <MEM0><MEM1><MEM2><MEM3><MEM4><MEM5><MEM6><MEM7><SEP>
        # assert 1==2
        # proper names for "eval" call, don't remove these lines
        docs = mem_tokens_str * self.generation_top_k
        question = query
        
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        # prompt_user = f"\n\nBackground:\n{docs}\n\nQuestion:{question}"
        prompt_user = f"{prompt_system}\n\nBackground:\n{docs}\n\nQuestion:{question}"
        
        # Prepare the messages with system and user roles
        # messages = [
        #     {"role": "system", "content": prompt_system},
        #     {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
        # ]

        # 构造单一用户角色的消息
        messages = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        # Attempt to apply the system role and catch if it's not supported
        try:
            # 将聊天消息格式化为模型可以理解的输入文本。这个函数通常会根据特定的模板将用户和助手的消息组合成一个统一的输入字符串。
            prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                # Apply template again without system role
                prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        return prompt

    # def blend_prompt_and_memory_tokens(self, query: str):
     
    #     mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token

    #     docs = mem_tokens_str * self.generation_top_k
    #     question = query
        
    #     prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
    #     prompt_user = f"\n\nBackground:\n{docs}\n\nQuestion:{question}"

    #     messages = [
    #         {"role": "system", "content": prompt_system},
    #         {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
    #     ]

    #     # 构造单一用户角色的消息
    #     messages = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

    #     # 将聊天消息格式化为模型可以理解的输入文本
    #     prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       

    #     return prompt


if __name__ == '__main__':
    cfg = COCOMConfig(decoder_model_name='mistralai/Mistral-7B-Instruct-v0.2',
                compr_model_name = "mistral_trimmed",
                compr_rate = 64,
                compr_n_layers = 5,
                compr_mlp_hidden_dim = 8096,
                compr_use_mlp = False, 
                lora = True, # lora on decoder (and decoder as compr)
                lora_compressor = True, # lora only on the compressor if it exists
                training_form = "both",
                load_adapters = True,
                kbtc_training = False,
                optimize_mem_tokens = True,
                different_mem_tokens = True,
                attn_implementation = 'flash_attention_2')
    
    cocom = COCOM(cfg)
    
    cocom.save_pretrained('test_ckpt')
    
    del cocom
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    cocom = COCOM.from_pretrained('test_ckpt')

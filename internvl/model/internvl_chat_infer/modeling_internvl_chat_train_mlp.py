# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         # new_weight = torch.rand(1, 4096)
#         # m.weight.copy_(new_weight)
#         nn.init.uniform_(m.weight, a=0.0, b=1.0)
#         nn.init.zeros_(m.bias)
#         print('m.weight',m.weight)
#         print('m.bias',m.bias)
        
class MLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 初始化线性层权重在 [0, 1] 之间
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print('m.weight1',m.weight)
                # with torch.no_grad():
                m.weight.data.uniform_(0.0, 1.0)
                print('m.weight2',m.weight)
                
                m.bias.data.uniform_(0.0, 1.0)
                print('m.bias',m.bias)
                    # if m.bias is not None:
                    #     nn.init.uniform_(m.bias, a=0.0, b=1.0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 自定义初始化函数：将权重初始化为 [0, 1) 之间的随机数
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         init.uniform_(m.weight, a=0.0, b=1.0)  # 权重初始化为 [0, 1) 随机数
#         init.zeros_(m.bias)  # 偏置初始化为 0

# # 创建模型
# mlp = MLP()


# class MLP(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.input_size = input_size
        
#         self.layers = nn.Sequential(
#             nn.Linear(self.input_size, 1024),
#             #nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 128),
#             #nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             #nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, 16),
#             #nn.ReLU(),
#             nn.Linear(16, 1)
#         )
        
#         # initial MLP param
#         for name, param in self.layers.named_parameters():
#             if 'weight' in name:
#                 nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
#             if 'bias' in name:
#                 nn.init.constant_(param, val=0)
        
#     def forward(self, input):
#         return self.layers(input)
    
class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        print('this model')
        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.mlpscore = MLP()
        # 使用自定义初始化函数对模型进行初始化
        # self.mlpscore.apply(init_weights)

        # 检查初始化结果
        print(self.mlpscore.fc1.weight)
        # self.mlpscore = nn.Sequential(
        #     # nn.LayerNorm(4096), 
        #     # nn.GELU(),
        #     nn.Linear(4096, 1),
            # nn.LayerNorm(256), 
            # nn.GELU(),
            # nn.Linear(1024, 256),
            # # nn.Linear(256, 1),
            # # nn.LayerNorm(256),
            # # # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(256, 64),
            # # nn.LayerNorm(64),
            # # # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()  # 输出在0-1之间
        # )
        # self.mlpscore.apply(init_weights)
        # for layer in self.mlpscore:
        #     if isinstance(layer, nn.Linear):
        #         print('layer4096',layer.weight)
        # nn.init.xavier_normal_(self.mlpscore[0].weight)
        # nn.init.zeros_(self.mlpscore[0].bias)
        # nn.init.xavier_normal_(self.mlpscore[4].weight)
        # nn.init.zeros_(self.mlpscore[4].bias)
        # nn.init.uniform_(self.mlpscore[0].weight, a=0.0, b=1.0)
        # # nn.init.normal_(self.mlpscore[0].weight, mean=0, std=4.1875 / (4096**0.5))
        # # nn.init.kaiming_normal_(self.mlpscore[1].weight, nonlinearity='relu')
        # nn.init.zeros_(self.mlpscore[0].bias)
        # print('maxweight',self.mlpscore[0].weight.max())
        # nn.init.xavier_normal_(self.mlpscore[3].weight)
        # nn.init.zeros_(self.mlpscore[3].bias)
        # nn.init.xavier_normal_(self.mlpscore[5].weight)
        # nn.init.zeros_(self.mlpscore[5].bias)
        # nn.init.xavier_normal_(self.mlpscore[7].weight)
        # nn.init.zeros_(self.mlpscore[7].bias)
        # self.mlpscore = nn.Sequential(
        #     # nn.Linear(4096, 1),
        #     nn.LayerNorm(4096),
        #     nn.Linear(4096, 1024),
        #     # nn.GELU(),
        #     # nn.Linear(4096, 1024),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(1024, 128),
        #     # nn.GELU(),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(128, 64),
        #     # nn.GELU(),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(64, 16),
        #     # nn.GELU(),
        #     # nn.ReLU(),
        #     nn.Linear(16, 1)
        # )
        # nn.init.xavier_normal_(self.mlpscore.weight)
        # nn.init.zeros_(self.mlpscore.bias)
        # print(self.mlpscore[1].weight)
        # print(self.mlpscore[1].bias)
        # nn.init.normal_(self.mlpscore[1].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[1].bias)
        # nn.init.normal_(self.mlpscore[2].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[2].bias)
        # nn.init.normal_(self.mlpscore[3].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[3].bias)
        # nn.init.normal_(self.mlpscore[4].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[4].bias)
        # nn.init.normal_(self.mlpscore[5].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[5].bias)
        # nn.init.normal_(self.mlpscore[1].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[1].bias)
        # nn.init.normal_(self.mlpscore[3].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[3].bias)
        # nn.init.normal_(self.mlpscore[5].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[5].bias)
        # nn.init.normal_(self.mlpscore[7].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[7].bias)
        # nn.init.normal_(self.mlpscore[9].weight,mean=0.0, std=1.0/(4096+1))
        # nn.init.zeros_(self.mlpscore[9].bias)
        # print(self.mlpscore[1].weight)
        # print(self.mlpscore[1].bias)
        # nn.init.xavier_normal_(model[2].weight)
        # nn.init.zeros_(model[2].bias)

        # nn.init.xavier_normal_(model[4].weight)
        # nn.init.zeros_(model[4].bias)

        # for name, param in self.mlpscore.named_parameters():
        #     print(name)
        #     if 'weight' in name:
        #         nn.init.normal_(param, mean=0.0, std=1.0/(4096+1))
        #     if 'bias' in name:
        #         nn.init.constant_(param, val=0)
        # torch.set_default_dtype(torch.bfloat16)
        # self.mlpscore = nn.Linear(5,1).bfloat16()
        # print('self.mlpscore.weight',self.mlpscore.weight)
        # print('self.mlpscore.bias',self.mlpscore.bias)
        # weight_tensor = torch.tensor([[1,0.75,0.5,0.25,0.]],dtype=torch.bfloat16)
        # bias_tensor = torch.tensor([0],dtype=torch.bfloat16)
        # from torch.nn.parameter import Parameter
        # self.mlpscore.weight = Parameter(weight_tensor)
        # self.mlpscore.bias = Parameter(bias_tensor)
        # print('self.mlpscore.weight',self.mlpscore.weight)
        # print('self.mlpscore.bias',self.mlpscore.bias)
        
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            mos: torch.FloatTensor,
            pixel_values: torch.FloatTensor,
            # pixel_values2: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(pixel_values.shape)
        # print(pixel_values2.shape)
        print('mos',mos)
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        # input_embeds2 = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        
        # vit_embeds2 = self.extract_feature(pixel_values2)
        # vit_embeds2 = vit_embeds2[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        # input_embeds2 = input_embeds2.reshape(B * N, C)

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            
        # try:
        #     input_embeds2[selected] = input_embeds2[selected] * 0.0 + vit_embeds2.reshape(-1, C)
        # except Exception as e:
        #     vit_embeds2 = vit_embeds2.reshape(-1, C)
        #     print(f'warning: {e}, input_embeds[selected].shape={input_embeds2[selected].shape}, '
        #           f'vit_embeds.shape={vit_embeds.shape}')
        #     n_token = selected.sum()
        #     input_embeds2[selected] = input_embeds2[selected] * 0.0 + vit_embeds2[:n_token]
        
        input_embeds = input_embeds.reshape(B, N, C)
        output_hidden_states = True
        # input_embeds2 = input_embeds2.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        # print(outputs)
        # print('output_hidden_states', outputs.hidden_states)
        last_hidden_states = outputs.hidden_states[-1]
        print('last_hidden_states', last_hidden_states[:,-4,:].shape)
        print(last_hidden_states[:,-4,:])
        input_tensor = last_hidden_states[:,-4,:]
        if torch.isnan(last_hidden_states[:,-4,:]).any():
            print("Input contains NaN values!")
            input_tensor = torch.nan_to_num(last_hidden_states[:,-4,:], nan=0.0, posinf=1e9, neginf=-1e9)
        print(input_tensor.mean())
        print(input_tensor.std())
        print(input_tensor.max())
        print(input_tensor.min())
        
        score1 = self.mlpscore(input_tensor)*1e-5
        print('score1',score1)
        print(self.mlpscore.fc1.weight)
        # print(score1.shape)
        # # preferential_ids_ = [9202,1811,6776,7989,4028]
        # ids_rate = logits[:,-4, preferential_ids_]
        
        # print('torch.softmax(ids_rate, -1)',torch.softmax(ids_rate, -1))
        # print('torch.softmax(ids_rate, -1)',torch.softmax(ids_rate, -1).bfloat16())
        # score1 = self.mlpscore(torch.softmax(ids_rate, -1).bfloat16())
        # print('self.mlpscore.weight',self.mlpscore.weight)
        # print('self.mlpscore.bias',self.mlpscore.bias)
        # print('video1:',score1)
        loss1_fct = nn.MSELoss(reduction = 'mean')
        print(mos.shape)
        score1 = score1.squeeze(1)
        print(score1.shape)
        loss1 = loss1_fct(score1, mos)
        
        # print('loss1',loss1)
        
    
        # weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).to(ids_rate.device)
        # score1 = torch.softmax(ids_rate, -1) @ weight_tensor
        
        # outputs2 = self.language_model(
        #     inputs_embeds=input_embeds2,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # logits2 = outputs2.logits
        
        # ids_rate2 = logits2[:,-4, preferential_ids_]
        # weight_tensor2 = torch.Tensor([1,0.75,0.5,0.25,0.]).to(ids_rate2.device)
        # score2 = torch.softmax(ids_rate2, -1) @ weight_tensor2
        
        # print('video2:',score2)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        print('loss',loss1)
        return CausalLMOutputWithPast(
            loss=loss1,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

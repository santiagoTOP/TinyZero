# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, # 定义每个节点上使用的GPU数量
                                            use_gpu=True, # 是否使用GPU
                                            max_colocate_count=1, # 定义每个资源池中可以同时放置的WorkerGroup数量
                                            name_prefix=resource_pool_name) # 定义资源池的名称
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        # Reward_final = Reward_original - β × KL(π || π_ref)
        # 用来调控β的值，β越大，KL惩罚越大，β越小，KL惩罚越小
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # 这段代码在运行时计算总训练步数，然后临时解除 OmegaConf 的结构保护
        # 将这个值注入到 Actor 和 Critic 的优化器配置中，供学习率调度器使用。
        # 注入完成后，配置重新变为只读，防止意外修改。
        OmegaConf.set_struct(self.config, True)  # 启动保护机制
        with open_dict(self.config):  # 临时解除保护，允许修改配置
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # 调用ActorRolloutWorker的generate_sequences方法，生成答案
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded) 
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """
        # ========== 第 1 步：准备类字典 ==========
        class_dict = {
            'actor_rollout': RayClassWithInitArgs(ActorRolloutRefWorker, ...),
            'critic': RayClassWithInitArgs(CriticWorker, ...),
            'ref': RayClassWithInitArgs(ActorRolloutRefWorker, ...)
        }

        # ========== 第 2 步：创建 WorkerDict 类 ==========
        worker_dict_cls = create_colocated_worker_cls(class_dict)
        # 内部创建了：
        # class WorkerDict(Worker):
        #     def __init__(self):
        #         self.worker_dict = {
        #             'actor_rollout': ActorRolloutRefWorker(...),
        #             'critic': CriticWorker(...),
        #             'ref': ActorRolloutRefWorker(...)
        #         }
        #     
        #     # 方法被 monkey-patch 上去：
        #     def actor_rollout__generate_sequences(self, data):
        #         return self.worker_dict['actor_rollout'].generate_sequences(data)
        #     
        #     def critic__compute_values(self, data):
        #         return self.worker_dict['critic'].compute_values(data)
        #     
        #     def ref__compute_ref_log_prob(self, data):
        #         return self.worker_dict['ref'].compute_ref_log_prob(data)

        # ========== 第 3 步：创建 WorkerGroup ==========
        wg_dict = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=worker_dict_cls)
        # 在每个 GPU 上实例化 WorkerDict：
        # GPU 0: WorkerDict 实例 (包含 actor_rollout, critic, ref)
        # GPU 1: WorkerDict 实例 (包含 actor_rollout, critic, ref)
        # GPU 2: WorkerDict 实例 (包含 actor_rollout, critic, ref)
        # GPU 3: WorkerDict 实例 (包含 actor_rollout, critic, ref)

        # ========== 第 4 步：Spawn 独立视图 ==========
        spawn_wg = wg_dict.spawn(prefix_set=['actor_rollout', 'critic', 'ref'])
        # spawn_wg = {
        #     'actor_rollout': WorkerGroup(只暴露 actor_rollout 方法),
        #     'critic': WorkerGroup(只暴露 critic 方法),
        #     'ref': WorkerGroup(只暴露 ref 方法)
        # }
        # 但它们都指向同一组 WorkerDict 实例！

        # ========== 第 5 步：使用 ==========
        actor_rollout_wg = spawn_wg['actor_rollout']
        actor_rollout_wg.generate_sequences(data)

        # 实际执行路径：
        # 1. actor_rollout_wg.generate_sequences(data)
        # 2. → wg_dict.execute_all('generate_sequences', data)
        # 3. → Ray RPC 到所有 GPU 的 WorkerDict 实例
        # 4. → WorkerDict.actor_rollout__generate_sequences(data)
        # 5. → self.worker_dict['actor_rollout'].generate_sequences(data)
        # 6. → ActorRolloutRefWorker.generate_sequences(data) ← 最终执行

        # ========== 第 6 步：保持引用 ==========
        self.wg_dicts.append(wg_dict)  # ← 保持对 WorkerDict 的引用
        
        """
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        # 创建资源池到类的映射，用于创建WorkerGroup
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout) # 获取ActorRollout可以使用的资源池
            # 包装ActorRollout类以及对应的初始化参数，用于创建WorkerGroup
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            # 在资源池中添加ActorRollout类
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        # create_colocated_worker_cls 的作用：
        # ✅ 合并多个 Worker 到同一进程：Actor、Rollout、Ref 共享一个进程
        # ✅ 节省 GPU 内存：共享模型权重，不需要多次加载
        # ✅ 减少通信开销：进程内通信比进程间通信快得多
        # ✅ 提高资源利用率：通过时间分片共享 GPU
        # 这是 FSDP 模式下的关键优化技术，让多个角色高效地共享同一个 GPU！
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            """
            具体方法的生成过程：
            
            第 569 行: worker_dict_cls = create_colocated_worker_cls(...)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📦 创建 WorkerDict 类
            📌 绑定到：类
            🏷️  方法名：带前缀 (critic__init_model)
            🔧 函数类型：简单转发函数
                def func(self, *args, **kwargs):
                    return self.worker_dict['critic'].init_model(*args, **kwargs)


            第 590 行: wg_dict = RayWorkerGroup(ray_cls_with_init=worker_dict_cls)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📦 创建 RayWorkerGroup 实例 (wg_dict)
            📌 绑定到：实例
            🏷️  方法名：带前缀 (critic__init_model)
            🔧 函数类型：完整代理函数 (func_generator)
                def func(*args, **kwargs):
                    args, kwargs = dispatch_fn(...)
                    output = execute_fn('critic__init_model', ...)
                    output = ray.get(output)
                    return collect_fn(...)


            第 610 行: spawn_wg = wg_dict.spawn(prefix_set=...)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📦 创建独立的 RayWorkerGroup 实例 (critic_wg, actor_rollout_wg, ...)
            📌 绑定到：新实例
            🏷️  方法名：无前缀 (init_model) ← 🔥 去除前缀！
            🔧 函数类型：完整代理函数 (func_generator)
                def func(*args, **kwargs):
                    args, kwargs = dispatch_fn(...)
                    output = execute_fn('init_model', ...)  # ← 注意：这里还是用原来的带前缀方法
                    output = ray.get(output)
                    return collect_fn(...)
            """
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # 创建 WorkerGroup
            """
            第 571 行: wg_dict = RayWorkerGroup(...)
                ↓
            RayWorkerGroup.__init__ (第 178 行)
                ↓
            第 203 行: self._bind_worker_method(...)  ← 🔥 第一次绑定（带前缀的方法）
                ↓
            第 591 行: spawn_wg = wg_dict.spawn(...)
                ↓
            spawn 方法中调用 from_detached (第 312 行)
                ↓
            from_detached 调用 RayWorkerGroup.__init__ (第 286 行)
                ↓
            第 203 行: self._bind_worker_method(...)  ← 🔥 第二次绑定（再次绑定所有方法）
                ↓
            _rebind_actor_methods (第 315 行)  ← 🔥 去掉前缀
                ↓
            第 621 行: self.critic_wg = all_wg['critic']  ← 获取最终的 critic_wg
            """
            # 生成独立的 WorkerGroup 引用
            # WorkerDict 内部结构
            # WorkerDict {
            #     worker_dict: {
            #         'actor_rollout': ActorRolloutRefWorker实例,
            #         'critic': CriticWorker实例,
            #         'ref': ActorRolloutRefWorker实例
            #     }
            # }
            # 但是，这些 worker 的方法被重命名了（添加了前缀）
            # WorkerDict 的方法
            # worker_dict.actor_rollout__generate_sequences()  # ← 有前缀
            # worker_dict.actor_rollout__update_actor()
            # worker_dict.critic__compute_values()             # ← 有前缀
            # worker_dict.ref__compute_ref_log_prob()          # ← 有前缀
            # 希望能这样调用
            # actor_rollout_wg.generate_sequences()  # ← 没有前缀
            # critic_wg.compute_values()             # ← 没有前缀
            # ref_wg.compute_ref_log_prob()          # ← 没有前缀
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys()) # ray/base.py
            # spawn_wg = {
            #     'actor_rollout': WorkerGroup(...),
            #     'ref': WorkerGroup(...)
            # }

            # spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys()) 的作用：
            # ✅ 创建独立视图：为每个 colocated worker 创建独立的 WorkerGroup 对象
            # ✅ 去掉方法前缀：actor_rollout__generate_sequences → generate_sequences
            # ✅ 保持共享进程：所有视图指向同一个底层 Ray Actor
            # ✅ 简化 API：使用方式与独立 WorkerGroup 一致
            # 这是一个代理模式（Proxy Pattern）的应用，让 colocated workers 的使用体验与独立 workers 完全一致！
            
            # 必须带前缀调用
            # wg_dict.execute_all('actor_rollout__generate_sequences', data)
            # wg_dict.execute_all('critic__compute_values', data)
            # wg_dict.execute_all('ref__compute_ref_log_prob', data)
            # 可以不带前缀调用
            # spawn_wg['actor_rollout'].execute_all('generate_sequences', data)
            # spawn_wg['critic'].execute_all('compute_values', data)
            # spawn_wg['ref'].execute_all('compute_ref_log_prob', data)

            # # 或者更简洁
            # actor_rollout_wg = spawn_wg['actor_rollout']
            # actor_rollout_wg.generate_sequences(data)  # ← 就像独立的 WorkerGroup
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            """
            时间点 1: 创建 WorkerGroup
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            wg_dict = RayWorkerGroup(...)
                ↓
            RayWorkerGroup.__init__
                ↓
            self._bind_worker_method(...)  ← 🔥 方法在这里被创建和绑定
                ↓
            遍历 Worker 类的所有方法
                ↓
            对于每个被 @register 装饰的方法:
                1. 调用 func_generator 创建代理函数
                2. setattr(self, 'init_model', func)  ← init_model 方法被添加到实例上
                ↓
            WorkerGroup 初始化完成
            此时 wg_dict.init_model 已经存在 ✅


            时间点 2: 调用方法（稍后）
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self.critic_wg.init_model()  ← 调用已经存在的方法
                ↓
            执行 func_generator 生成的代理函数
                ↓
            1. dispatch_fn(...)  # 分发参数
            2. execute_fn(...)   # 执行远程调用
            3. ray.get(...)      # 等待结果
            4. collect_fn(...)   # 收集结果

            # 步骤 1: 你写的代码
            self.critic_wg.init_model()

            # 步骤 2: Python 解释器查找 init_model 属性
            # 找到：init_model = func_generator(...) 返回的 func

            # 步骤 3: 调用这个 func 函数
            # 进入 base.py:38-44 的 func 函数体

            # 步骤 4: 执行 func 内部的逻辑
            def func(*args, **kwargs):  # ← 你在这里！(base.py:38)
                # 第 39 行：分发参数
                args, kwargs = dispatch_fn(self, *args, **kwargs)
                
                # 第 40 行：执行远程调用
                # execute_fn 是 self.execute_all
                # method_name 是 'init_model'（通过闭包捕获）
                output = execute_fn('init_model', *args, **kwargs)
                
                # 第 41-42 行：等待结果
                if blocking:
                    output = ray.get(output)
                
                # 第 43 行：收集结果
                output = collect_fn(self, output)
                
                # 第 44 行：返回
                return output
            
            所有被 @register 装饰器装饰的方法都会通过 func_generator 生成代理函数。
            """
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)  
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, # 原本的样本序列长度列表
                                                    partitions=global_partition_lst, # 平衡后的样本序列，每个partition是一个列表，列表中是样本的索引
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):  # 全部的数据训练15轮
            for batch_dict in self.train_dataloader:  # 每次训练一个batch
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict) # 将batch_dict转换为DataProto对象

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids']) # 将batch中的input_ids, attention_mask, position_ids提取出来，此时原始的batch中不再包含这些key

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # 使用actor_rollout_wg生成响应

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    """
                    假设 n=3，batch_size=2

                    原始 batch:
                    ├─ prompt_0: "What is 2+2?"
                    │  └─ uid: "uuid-0"
                    └─ prompt_1: "What is 3+3?"
                    └─ uid: "uuid-1"

                    ↓ generate_sequences (vLLM 生成 n=3 个 responses)

                    gen_batch_output (6 个 responses):
                    ├─ prompt_0 → response: "4" (reward: 1.0)
                    ├─ prompt_0 → response: "5" (reward: 0.0)
                    ├─ prompt_0 → response: "4" (reward: 1.0)
                    ├─ prompt_1 → response: "6" (reward: 1.0)
                    ├─ prompt_1 → response: "7" (reward: 0.0)
                    └─ prompt_1 → response: "6" (reward: 1.0)

                    ↓ batch.repeat(repeat_times=3, interleave=True)

                    复制后的 batch (6 个):
                    ├─ uid: "uuid-0"  ──┐
                    ├─ uid: "uuid-0"    ├─ 同一组，计算相对优势
                    ├─ uid: "uuid-0"  ──┘
                    ├─ uid: "uuid-1"  ──┐
                    ├─ uid: "uuid-1"    ├─ 同一组，计算相对优势
                    └─ uid: "uuid-1"  ──┘

                    ↓ compute_grpo_outcome_advantage

                    GRPO 优势计算:
                    - uuid-0 组: rewards=[1.0, 0.0, 1.0] → mean=0.67, std=0.47
                    - response 0: advantage = (1.0-0.67)/0.47 = +0.70 ✅ 鼓励
                    - response 1: advantage = (0.0-0.67)/0.47 = -1.43 ❌ 惩罚
                    - response 2: advantage = (1.0-0.67)/0.47 = +0.70 ✅ 鼓励
                    """
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # 复制其他数据以匹配生成的 responses 数量，在vllm采样的时候，一个prompt可能对应多个response，GRPO
                    
                    """
                    原始 batch (从 dataloader 加载):
                    ├─ input_ids: [prompt tokens]
                    ├─ attention_mask: [prompt mask]
                    ├─ position_ids: [prompt positions]
                    └─ 其他数据 (如 labels, 等)

                    ↓ pop(['input_ids', 'attention_mask', 'position_ids'])

                    batch (pop 后):                      gen_batch (被 pop 出来):
                    ├─ 其他数据                          ├─ input_ids: [prompt tokens]
                                                        ├─ attention_mask: [prompt mask]
                                                        └─ position_ids: [prompt positions]

                    ↓ generate_sequences(gen_batch)

                                                        gen_batch_output (生成后):
                                                        ├─ prompts: [prompt tokens]
                                                        ├─ responses: [response tokens]
                                                        ├─ input_ids: [prompt + response] ⭐ 变化了！
                                                        ├─ attention_mask: [full mask] ⭐ 变化了！
                                                        └─ position_ids: [full positions] ⭐ 变化了！

                    ↓ batch.repeat(n) + batch.union(gen_batch_output)

                    batch (最终):
                    ├─ 其他数据 (重复 n 次)
                    ├─ prompts: [prompt tokens]
                    ├─ responses: [response tokens]
                    ├─ input_ids: [prompt + response] ⭐ 新的完整序列
                    ├─ attention_mask: [full mask] ⭐ 新的完整 mask
                    └─ position_ids: [full positions] ⭐ 新的完整 positions
                    """
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics) # 平衡每个dp rank的有效token数量，确保每个rank都有相同数量的有效token

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:  # 使用模型进行奖励
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

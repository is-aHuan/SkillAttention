import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)


class SkillDeltaAttention(nn.Module):
    """
    技能增量注意力模块 - 全局训练版本
    结合VQ codebook技能选择和增量交叉注意力机制
    """
    
    def __init__(
        self,
        codebook,
        config,
        hidden_size: int,
        skill_pool_size: int,
        skill_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        delta_rank: int = 16,
        alpha: float = 1.0
    ):
        super().__init__()
        self.option_codebook = codebook

        self.hidden_size = hidden_size
        self.skill_pool_size = skill_pool_size
        self.skill_dim = skill_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.delta_rank = delta_rank
        self.alpha = alpha
        
        self.key = nn.Linear(skill_dim, self.all_head_size)
        self.value = nn.Linear(skill_dim, self.all_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        
        
        # 增量注意力参数（低秩分解）
        self.delta_q_down = nn.Linear(hidden_size, delta_rank, bias=False)
        self.delta_q_up = nn.Linear(delta_rank, hidden_size, bias=False)
        
        self.delta_k_down = nn.Linear(hidden_size, delta_rank, bias=False)
        self.delta_k_up = nn.Linear(delta_rank, hidden_size, bias=False)
        
        self.delta_v_down = nn.Linear(hidden_size, delta_rank, bias=False)
        self.delta_v_up = nn.Linear(delta_rank, hidden_size, bias=False)
        
        # 技能感知的增量参数
        self.skill_aware_q = nn.Linear(skill_dim, delta_rank, bias=False)
        self.skill_aware_k = nn.Linear(skill_dim, delta_rank, bias=False)
        self.skill_aware_v = nn.Linear(skill_dim, delta_rank, bias=False)
        
        # 门控机制
        self.skill_gate = nn.Sequential(
            nn.Linear(hidden_size + skill_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        # 标准初始化注意力参数
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        # 技能codebook初始化
        nn.init.normal_(self.skill_codebook, std=0.02)
        nn.init.normal_(self.skill_pool, std=0.02)
        
        # 增量参数初始化为较小值
        for module in [self.delta_q_down, self.delta_q_up, 
                      self.delta_k_down, self.delta_k_up,
                      self.delta_v_down, self.delta_v_up,
                      self.skill_aware_q, self.skill_aware_k, self.skill_aware_v]:
            nn.init.normal_(module.weight, std=0.01)
    
    
    def compute_skill_weighted_deltas(self, x: torch.Tensor, selected_skills: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算技能加权的增量参数
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            selected_skills: [batch_size, skill_dim]
            
        Returns:
            delta_params: 包含增量Q,K,V的字典
        """
        # 计算技能感知的增量
        skill_q_delta = self.skill_aware_q(selected_skills)  # [batch_size, delta_rank]
        skill_k_delta = self.skill_aware_k(selected_skills)  # [batch_size, delta_rank]
        skill_v_delta = self.skill_aware_v(selected_skills)  # [batch_size, delta_rank]
        
        # 计算输入相关的增量
        input_q_delta = self.delta_q_down(x)  # [batch_size, seq_len, delta_rank]
        input_k_delta = self.delta_k_down(x)  # [batch_size, seq_len, delta_rank]
        input_v_delta = self.delta_v_down(x)  # [batch_size, seq_len, delta_rank]
        
        # 组合技能感知和输入相关的增量
        combined_q_delta = input_q_delta + skill_q_delta.unsqueeze(1)
        combined_k_delta = input_k_delta + skill_k_delta.unsqueeze(1)
        combined_v_delta = input_v_delta + skill_v_delta.unsqueeze(1)
        
        # 投影到原始维度
        delta_q = self.delta_q_up(combined_q_delta) * self.alpha
        delta_k = self.delta_k_up(combined_k_delta) * self.alpha
        delta_v = self.delta_v_up(combined_v_delta) * self.alpha
        
        return {
            'delta_q': delta_q,
            'delta_k': delta_k,
            'delta_v': delta_v
        }
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            output: 包含注意力输出和损失的字典
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. 从codebook选择技能
        selected_skills, skill_indices, quantization_loss = self.select_skill_from_codebook(x)
        
        # 2. 计算技能加权的增量参数
        delta_params = self.compute_skill_weighted_deltas(x, selected_skills)
        
        # 3. 计算基础Q, K, V
        base_q = self.q_proj(x)
        base_k = self.k_proj(x)
        base_v = self.v_proj(x)
        
        # 4. 添加技能增量
        q = base_q + delta_params['delta_q']
        k = base_k + delta_params['delta_k']
        v = base_v + delta_params['delta_v']
        
        # 5. 重塑为多头注意力格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 6. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            mask_value = -1e9 if q.dtype == torch.float32 else -1e4
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # 8. 输出投影
        output = self.out_proj(attn_output)
        
        # 9. 技能门控机制
        gate_input = torch.cat([x.mean(dim=1), selected_skills], dim=-1)
        gate = self.skill_gate(gate_input).unsqueeze(1)
        
        # 10. 残差连接和层归一化
        output = self.layer_norm(x + gate * output)
        
        return {
            'output': output,
            'selected_skills': selected_skills,
            'skill_indices': skill_indices,
            'quantization_loss': quantization_loss,
            'attention_weights': attn_weights,
            'gate_weights': gate
        }


class SkillDeltaBlock(nn.Module):
    """
    包含技能增量注意力的完整Transformer块
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        skill_pool_size: int, 
        skill_dim: int, 
        ffn_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        if ffn_dim is None:
            ffn_dim = hidden_size * 4
            
        self.skill_attention = SkillDeltaAttention(
            hidden_size=hidden_size,
            skill_pool_size=skill_pool_size,
            skill_dim=skill_dim,
            **kwargs
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(kwargs.get('dropout', 0.1))
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 技能增量注意力
        attn_out = self.skill_attention(x, attention_mask)
        x = attn_out['output']
        
        # Feed-Forward Network with residual connection
        x = self.ffn_norm(x + self.ffn(x))
        
        # 更新输出
        attn_out['output'] = x
        return attn_out


class MultiSkillDeltaAttention(nn.Module):
    """
    多层技能增量注意力模块
    """
    
    def __init__(
        self,
        hidden_size: int,
        skill_pool_size: int,
        skill_dim: int,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SkillDeltaBlock(
                hidden_size=hidden_size,
                skill_pool_size=skill_pool_size,
                skill_dim=skill_dim,
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        all_quantization_losses = []
        all_skill_indices = []
        all_gate_weights = []
        
        for layer in self.layers:
            layer_out = layer(x, attention_mask)
            x = layer_out['output']
            
            all_quantization_losses.append(layer_out['quantization_loss'])
            all_skill_indices.append(layer_out['skill_indices'])
            all_gate_weights.append(layer_out['gate_weights'])
        
        # 汇总输出
        total_quantization_loss = torch.stack(all_quantization_losses).mean()
        
        return {
            'output': x,
            'quantization_loss': total_quantization_loss,
            'all_skill_indices': all_skill_indices,
            'all_gate_weights': all_gate_weights,
            'final_skill_indices': all_skill_indices[-1],
            'final_gate_weights': all_gate_weights[-1]
        }
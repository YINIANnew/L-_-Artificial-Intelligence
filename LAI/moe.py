import math
import json

import sys
sys.path.append(sys.path[0])

from L_attention import Tensor, nn
from L_managment import AIConfigManager

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation='relu'):
        super(ExpertNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(self.activation)

            in_dim = out_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers
        }


class GatingMechanism(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1):
        super(GatingMechanism, self).__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate_weight = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        gate_scores = self.gate_weight(x)
        gate_weights = gate_scores.softmax(dim=-1)

        if self.top_k >= self.num_experts:
            selected_experts = Tensor([[i for i in range(self.num_experts)] for _ in range(x.shape[0])])
            top_k_weights = gate_weights
        else:
            print(f"gate_weights type: {type(gate_weights)}")
            print(f"gate_weights shape: {gate_weights.shape}")
            
            batch_size = len(gate_weights.data)
            num_experts = len(gate_weights.data[0])
            
            top_k_weights = []
            selected_experts = []
            
            for row in gate_weights.data:
                sorted_indices = sorted(range(num_experts), key=lambda j: row[j], reverse=True)
                top_indices = sorted_indices[:self.top_k]
                top_values = [row[j] for j in top_indices]
                
                sum_values = sum(top_values)
                if sum_values > 0:
                    top_values = [v / sum_values for v in top_values]
                
                top_k_weights.append(top_values)
                selected_experts.append(top_indices)
            
            top_k_weights = Tensor(top_k_weights)
            selected_experts = Tensor(selected_experts)
            print(f"top_k_weights type: {type(top_k_weights)}")
            print(f"selected_experts type: {type(selected_experts)}")

        return top_k_weights, selected_experts

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'num_experts': self.num_experts,
            'top_k': self.top_k
        }


class RoutingAlgorithm(nn.Module):
    def __init__(self, num_experts, routing_strategy='sparse', temperature=1.0):
        super(RoutingAlgorithm, self).__init__()

        self.num_experts = num_experts
        self.routing_strategy = routing_strategy
        self.temperature = temperature

    def route(self, gate_weights, selected_experts, x):
        try:
            batch_size = x.shape[0]
            print(f"batch_size: {batch_size}")
            print(f"num_experts: {self.num_experts}")
            print(f"selected_experts shape: {selected_experts.shape}")
            print(f"gate_weights shape: {gate_weights.shape}")

            routed_weights = Tensor([[0.0 for _ in range(self.num_experts)] for _ in range(batch_size)])
            print(f"routed_weights shape: {routed_weights.shape}")

            max_i = min(selected_experts.shape[1], gate_weights.shape[1])
            print(f"max_i: {max_i}")
            
            for i in range(max_i):
                print(f"Processing i: {i}")
                expert_idx = selected_experts[:, i]
                weight = gate_weights[:, i]
                print(f"expert_idx type: {type(expert_idx)}")
                print(f"weight type: {type(weight)}")
                
                for j in range(batch_size):
                    print(f"Processing j: {j}")
                    if isinstance(expert_idx, Tensor):
                        expert_id = expert_idx[j]
                    else:
                        expert_id = expert_idx[j]
                    
                    if isinstance(weight, Tensor):
                        weight_val = weight[j]
                    else:
                        weight_val = weight[j]
                    
                    if expert_id < routed_weights.shape[1]:
                        routed_weights[j, expert_id] += weight_val
                    else:
                        print(f"expert_id {expert_id} out of range for routed_weights shape {routed_weights.shape}")

            if self.temperature != 1.0:
                routed_weights = routed_weights / self.temperature

            return routed_weights
        except Exception as e:
            print(f"Error in route method: {e}")
            import traceback
            traceback.print_exc()
            return Tensor([[0.0 for _ in range(self.num_experts)] for _ in range(x.shape[0])])

    def get_config(self):
        return {
            'num_experts': self.num_experts,
            'routing_strategy': self.routing_strategy,
            'temperature': self.temperature
        }


class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, alpha=0.1):
        super(LoadBalancingLoss, self).__init__()

        self.num_experts = num_experts
        self.alpha = alpha

    def forward(self, gate_weights, selected_experts, batch_size):
        expert_counts = Tensor([0.0 for _ in range(self.num_experts)])

        for i in range(selected_experts.shape[1]):
            expert_idx = selected_experts[:, i]
            for j in range(batch_size):
                if isinstance(expert_idx, Tensor):
                    if isinstance(expert_idx[j], Tensor):
                        expert_id = expert_idx[j].data
                    else:
                        expert_id = expert_idx[j]
                else:
                    expert_id = expert_idx[j]
                expert_counts[expert_id] += 1.0

        f_i = expert_counts / batch_size

        p_i = Tensor([0.0 for _ in range(self.num_experts)])

        for i in range(selected_experts.shape[1]):
            expert_idx = selected_experts[:, i]
            weight = gate_weights[:, i]
            for j in range(batch_size):
                if isinstance(expert_idx, Tensor):
                    if isinstance(expert_idx[j], Tensor):
                        expert_id = expert_idx[j].data
                    else:
                        expert_id = expert_idx[j]
                else:
                    expert_id = expert_idx[j]
                
                if isinstance(weight, Tensor):
                    if isinstance(weight[j], Tensor):
                        weight_val = weight[j].data
                    else:
                        weight_val = weight[j]
                else:
                    weight_val = weight[j]
                
                p_i[expert_id] += weight_val

        p_i = p_i / expert_counts.clamp(min=1)

        loss = self.num_experts * p_i.dot(f_i)

        return self.alpha * loss

    def get_config(self):
        return {
            'num_experts': self.num_experts,
            'alpha': self.alpha
        }


class MoE(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, num_experts=8, hidden_dim=256, expert_layers=2,
                 top_k=2, routing_strategy='sparse', load_balancing_alpha=0.1, config_file=None):
        super(MoE, self).__init__()

        if config_file:
            self.config_manager = AIConfigManager(config_file)
            config = self.config_manager.get_config()
            self.input_dim = config.get('input_dim', input_dim)
            self.output_dim = config.get('output_dim', output_dim)
            self.num_experts = config.get('num_experts', num_experts)
            self.hidden_dim = config.get('hidden_dim', hidden_dim)
            self.expert_layers = config.get('expert_layers', expert_layers)
            self.top_k = config.get('top_k', top_k)
            routing_strategy = config.get('routing_strategy', routing_strategy)
            load_balancing_alpha = config.get('load_balancing_alpha', load_balancing_alpha)
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_experts = num_experts
            self.hidden_dim = hidden_dim
            self.expert_layers = expert_layers
            self.top_k = top_k
            self.config_manager = None

        self.experts = []
        for _ in range(self.num_experts):
            self.experts.append(ExpertNetwork(self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers))

        self.gating = GatingMechanism(self.input_dim, self.num_experts, self.top_k)
        self.router = RoutingAlgorithm(self.num_experts, routing_strategy)
        self.load_balancer = LoadBalancingLoss(self.num_experts, load_balancing_alpha)

    def forward(self, x, return_loss=False):
        batch_size = x.shape[0]

        gate_weights, selected_experts = self.gating(x)
        routed_weights = self.router.route(gate_weights, selected_experts, x)
        print(f"routed_weights type: {type(routed_weights)}")
        print(f"routed_weights shape: {routed_weights.shape}")
        print(f"num_experts: {self.num_experts}")
        print(f"experts length: {len(self.experts)}")

        output = Tensor([[0.0 for _ in range(self.output_dim)] for _ in range(batch_size)])

        for i, expert in enumerate(self.experts):
            if not callable(expert):
                raise TypeError(f"Expert {i} is not callable: {type(expert)}")
            
            expert_output = expert(x)

            if self.top_k < self.num_experts:
                if i < routed_weights.shape[1]:
                    expert_weight = routed_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor([[0.0] for _ in range(batch_size)])
            else:
                if i < gate_weights.shape[1]:
                    expert_weight = gate_weights[:, i].unsqueeze(1)
                    for j in range(selected_experts.shape[1]):
                        match = False
                        for k in range(batch_size):
                            if k < selected_experts.shape[0] and j < selected_experts.shape[1]:
                                if selected_experts[k][j] == i:
                                    match = True
                                    break
                        if match:
                            expert_weight = gate_weights[:, j].unsqueeze(1)
                            break
                else:
                    expert_weight = Tensor([[0.0] for _ in range(batch_size)])

            output = output + expert_output * expert_weight

        if return_loss:
            load_balance_loss = self.load_balancer(gate_weights, selected_experts, batch_size)
            if isinstance(load_balance_loss, Tensor):
                load_balance_loss = load_balance_loss.data
            return output, load_balance_loss

        return output

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'hidden_dim': self.hidden_dim,
            'expert_layers': self.expert_layers,
            'top_k': self.top_k,
            'routing_strategy': self.router.routing_strategy,
            'load_balancing_alpha': self.load_balancer.alpha
        }

    def load_balancing_loss(self, gate_weights, selected_experts, batch_size):
        return self.load_balancer(gate_weights, selected_experts, batch_size)


class MoELoadingBalancer:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.expert_usage = [0.0 for _ in range(num_experts)]
        self.total_routing_count = 0

    def update(self, selected_experts, batch_size):
        for i in range(selected_experts.shape[1]):
            expert_idx = selected_experts[:, i]
            for j in range(batch_size):
                if isinstance(expert_idx, Tensor):
                    if isinstance(expert_idx[j], Tensor):
                        expert_id = expert_idx[j].data
                    else:
                        expert_id = expert_idx[j]
                else:
                    expert_id = expert_idx[j]
                self.expert_usage[expert_id] += 1.0

        self.total_routing_count += batch_size

    def get_usage_stats(self):
        usage_percentages = [0.0 for _ in range(self.num_experts)]
        if self.total_routing_count > 0:
            for i in range(self.num_experts):
                usage_percentages[i] = (self.expert_usage[i] / self.total_routing_count) * 100

        import numpy as np
        return {
            'expert_usage': self.expert_usage,
            'usage_percentages': usage_percentages,
            'total_routing_count': self.total_routing_count,
            'std': float(np.std(usage_percentages)),
            'is_balanced': float(np.std(usage_percentages)) < 10.0
        }

    def reset(self):
        self.expert_usage = [0.0 for _ in range(self.num_experts)]
        self.total_routing_count = 0


class DynamicMoE(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, min_experts=4, max_experts=16, 
                 hidden_dim=256, expert_layers=2, top_k=2, 
                 routing_strategy='sparse', load_balancing_alpha=0.1, config_file=None):
        super(DynamicMoE, self).__init__()
        
        if config_file:
            self.config_manager = AIConfigManager(config_file)
            config = self.config_manager.get_config()
            self.input_dim = config.get('input_dim', input_dim)
            self.output_dim = config.get('output_dim', output_dim)
            self.min_experts = config.get('min_experts', min_experts)
            self.max_experts = config.get('max_experts', max_experts)
            self.hidden_dim = config.get('hidden_dim', hidden_dim)
            self.expert_layers = config.get('expert_layers', expert_layers)
            self.top_k = config.get('top_k', top_k)
            routing_strategy = config.get('routing_strategy', routing_strategy)
            load_balancing_alpha = config.get('load_balancing_alpha', load_balancing_alpha)
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.min_experts = min_experts
            self.max_experts = max_experts
            self.hidden_dim = hidden_dim
            self.expert_layers = expert_layers
            self.top_k = top_k
            self.config_manager = None
        
        self.current_experts = self.min_experts
        
        self.experts = []
        for _ in range(self.current_experts):
            self.experts.append(ExpertNetwork(self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers))
        
        self.gating = GatingMechanism(self.input_dim, self.current_experts, self.top_k)
        self.router = RoutingAlgorithm(self.current_experts, routing_strategy)
        self.load_balancer = LoadBalancingLoss(self.current_experts, load_balancing_alpha)
        self.expert_evaluator = nn.Linear(self.output_dim, 1)
        self.expert_usage = [0.0 for _ in range(self.max_experts)]
        
    def forward(self, x, return_loss=False):
        batch_size = x.shape[0]
        
        gate_weights, selected_experts = self.gating(x)
        routed_weights = self.router.route(gate_weights, selected_experts, x)
        print(f"routed_weights type: {type(routed_weights)}")
        print(f"routed_weights shape: {routed_weights.shape}")
        print(f"current_experts: {self.current_experts}")
        print(f"experts length: {len(self.experts)}")
        
        output = Tensor([[0.0 for _ in range(self.output_dim)] for _ in range(batch_size)])
        
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            
            if self.top_k < self.current_experts:
                if i < routed_weights.shape[1]:
                    expert_weight = routed_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor([[0.0] for _ in range(batch_size)])
            else:
                if i < gate_weights.shape[1]:
                    expert_weight = gate_weights[:, i].unsqueeze(1)
                    for j in range(selected_experts.shape[1]):
                        match = False
                        for k in range(batch_size):
                            if k < selected_experts.shape[0] and j < selected_experts.shape[1]:
                                if selected_experts[k][j] == i:
                                    match = True
                                    break
                        if match:
                            expert_weight = gate_weights[:, j].unsqueeze(1)
                            break
                else:
                    expert_weight = Tensor([[0.0] for _ in range(batch_size)])
            
            output = output + expert_output * expert_weight
        
        self._update_expert_usage(selected_experts, batch_size)
        
        if return_loss:
            load_balance_loss = self.load_balancer(gate_weights, selected_experts, batch_size)
            if isinstance(load_balance_loss, Tensor):
                load_balance_loss = load_balance_loss.data
            return output, load_balance_loss
        
        return output
    
    def _update_expert_usage(self, selected_experts, batch_size):
        for i in range(selected_experts.shape[1]):
            expert_idx = selected_experts[:, i]
            for j in range(batch_size):
                if isinstance(expert_idx, Tensor):
                    if isinstance(expert_idx[j], Tensor):
                        expert_id = expert_idx[j].data
                    else:
                        expert_id = expert_idx[j]
                else:
                    expert_id = expert_idx[j]
                self.expert_usage[expert_id] += 1.0
    
    def evaluate_experts(self, x):
        expert_scores = []
        for expert in self.experts:
            output = expert(x)
            mean_score = self.expert_evaluator(output).mean()
            if isinstance(mean_score, Tensor):
                score = mean_score.data
            else:
                score = mean_score
            expert_scores.append(score)
        return expert_scores
    
    def adjust_experts(self, x):
        expert_scores = self.evaluate_experts(x)
        
        sorted_indices = sorted(range(len(expert_scores)), key=lambda i: expert_scores[i], reverse=True)
        
        import numpy as np
        if len(self.experts) < self.max_experts and np.std(expert_scores) > 0.1:
            new_expert = ExpertNetwork(self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers)
            best_expert = self.experts[sorted_indices[0]]
            self.experts.append(new_expert)
            self.current_experts += 1
            
            self.gating = GatingMechanism(self.input_dim, self.current_experts, self.top_k)
            self.router = RoutingAlgorithm(self.current_experts, self.router.routing_strategy)
            self.load_balancer = LoadBalancingLoss(self.current_experts, self.load_balancer.alpha)
            
            print(f"Added new expert. Current experts: {self.current_experts}")
        
        elif len(self.experts) > self.min_experts and np.std(expert_scores) < 0.05:
            worst_idx = sorted_indices[-1]
            self.experts.pop(worst_idx)
            self.current_experts -= 1
            
            self.gating = GatingMechanism(self.input_dim, self.current_experts, self.top_k)
            self.router = RoutingAlgorithm(self.current_experts, self.router.routing_strategy)
            self.load_balancer = LoadBalancingLoss(self.current_experts, self.load_balancer.alpha)
            
            print(f"Removed worst expert. Current experts: {self.current_experts}")
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'min_experts': self.min_experts,
            'max_experts': self.max_experts,
            'current_experts': self.current_experts,
            'hidden_dim': self.hidden_dim,
            'expert_layers': self.expert_layers,
            'top_k': self.top_k,
            'routing_strategy': self.router.routing_strategy,
            'load_balancing_alpha': self.load_balancer.alpha
        }
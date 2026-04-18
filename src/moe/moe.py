import numpy as np

from ..core.tensor import Tensor
from ..core.nn import nn
from ..config.config import AIConfigManager


class ExpertNetwork(nn.Module):

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, activation="relu"
    ):
        super(ExpertNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
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
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
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
            selected_experts = Tensor(
                np.array(
                    [[i for i in range(self.num_experts)] for _ in range(x.shape[0])]
                )
            )
            top_k_weights = gate_weights
        else:
            top_k_weights, selected_experts = gate_weights.topk(self.top_k, dim=-1)

        return top_k_weights, selected_experts

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


class RoutingAlgorithm(nn.Module):

    def __init__(self, num_experts, routing_strategy="sparse", temperature=1.0):
        super(RoutingAlgorithm, self).__init__()

        self.num_experts = num_experts
        self.routing_strategy = routing_strategy
        self.temperature = temperature

    def route(self, gate_weights, selected_experts, x):
        try:
            batch_size = x.shape[0]
            top_k = selected_experts.shape[1]

            routed_weights = np.zeros((batch_size, self.num_experts))

            expert_indices = selected_experts.data
            weight_values = gate_weights.data

            batch_indices = np.arange(batch_size)[:, None].repeat(top_k, axis=1)

            valid_mask = expert_indices < self.num_experts

            np.add.at(
                routed_weights,
                (batch_indices[valid_mask], expert_indices[valid_mask]),
                weight_values[valid_mask],
            )

            if self.temperature != 1.0:
                routed_weights = routed_weights / self.temperature

            return Tensor(routed_weights)
        except Exception as e:
            print(f"Error in route method: {e}")
            import traceback

            traceback.print_exc()
            return Tensor(np.zeros((x.shape[0], self.num_experts)))

    def get_config(self):
        return {
            "num_experts": self.num_experts,
            "routing_strategy": self.routing_strategy,
            "temperature": self.temperature,
        }


class LoadBalancingLoss(nn.Module):

    def __init__(self, num_experts, alpha=0.1):
        super(LoadBalancingLoss, self).__init__()

        self.num_experts = num_experts
        self.alpha = alpha

    def forward(self, gate_weights, selected_experts, batch_size):
        expert_indices = selected_experts.data
        weight_values = gate_weights.data

        valid_mask = expert_indices < self.num_experts

        expert_counts = np.zeros(self.num_experts)
        np.add.at(expert_counts, expert_indices[valid_mask], 1.0)
        f_i = expert_counts / batch_size

        p_i = np.zeros(self.num_experts)
        np.add.at(p_i, expert_indices[valid_mask], weight_values[valid_mask])
        p_i = p_i / np.maximum(expert_counts, 1)

        loss = self.num_experts * np.dot(p_i, f_i)

        return self.alpha * Tensor(loss)

    def get_config(self):
        return {"num_experts": self.num_experts, "alpha": self.alpha}


class MoE(nn.Module):

    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        num_experts=8,
        hidden_dim=256,
        expert_layers=2,
        top_k=2,
        routing_strategy="sparse",
        load_balancing_alpha=0.1,
        config_file=None,
    ):
        super(MoE, self).__init__()

        if config_file:
            self.config_manager = AIConfigManager(config_file)
            config = self.config_manager.get_config()
            self.input_dim = config.get("input_dim", input_dim)
            self.output_dim = config.get("output_dim", output_dim)
            self.num_experts = config.get("num_experts", num_experts)
            self.hidden_dim = config.get("hidden_dim", hidden_dim)
            self.expert_layers = config.get("expert_layers", expert_layers)
            self.top_k = config.get("top_k", top_k)
            routing_strategy = config.get("routing_strategy", routing_strategy)
            load_balancing_alpha = config.get(
                "load_balancing_alpha", load_balancing_alpha
            )
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
            self.experts.append(
                ExpertNetwork(
                    self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers
                )
            )

        self.gating = GatingMechanism(self.input_dim, self.num_experts, self.top_k)

        self.router = RoutingAlgorithm(self.num_experts, routing_strategy)

        self.load_balancer = LoadBalancingLoss(self.num_experts, load_balancing_alpha)

    def forward(self, x, return_loss=False):
        batch_size = x.shape[0]

        gate_weights, selected_experts = self.gating(x)

        routed_weights = self.router.route(gate_weights, selected_experts, x)

        output = Tensor(np.zeros((batch_size, self.output_dim)))

        for i, expert in enumerate(self.experts):
            if not callable(expert):
                raise TypeError(f"Expert {i} is not callable: {type(expert)}")

            expert_output = expert(x)

            if self.top_k < self.num_experts:
                if i < routed_weights.shape[1]:
                    expert_weight = routed_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor(np.zeros((batch_size, 1)))
            else:
                if i < gate_weights.shape[1]:
                    expert_weight = gate_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor(np.zeros((batch_size, 1)))

            output = output + expert_output * expert_weight

        if return_loss:
            load_balance_loss = self.load_balancer(
                gate_weights, selected_experts, batch_size
            )
            if isinstance(load_balance_loss, Tensor):
                load_balance_loss = load_balance_loss.data
            return output, load_balance_loss

        return output

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_experts": self.num_experts,
            "hidden_dim": self.hidden_dim,
            "expert_layers": self.expert_layers,
            "top_k": self.top_k,
            "routing_strategy": self.router.routing_strategy,
            "load_balancing_alpha": self.load_balancer.alpha,
        }

    def load_balancing_loss(self, gate_weights, selected_experts, batch_size):
        return self.load_balancer(gate_weights, selected_experts, batch_size)


class MoELoadingBalancer:

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.expert_usage = np.zeros(num_experts)
        self.total_routing_count = 0

    def update(self, selected_experts, batch_size):
        expert_indices = selected_experts.data

        valid_mask = expert_indices < self.num_experts

        np.add.at(self.expert_usage, expert_indices[valid_mask], 1.0)

        self.total_routing_count += batch_size

    def get_usage_stats(self):
        usage_percentages = np.zeros(self.num_experts)
        if self.total_routing_count > 0:
            usage_percentages = (self.expert_usage / self.total_routing_count) * 100

        return {
            "expert_usage": self.expert_usage.tolist(),
            "usage_percentages": usage_percentages.tolist(),
            "total_routing_count": self.total_routing_count,
            "std": float(np.std(usage_percentages)),
            "is_balanced": float(np.std(usage_percentages)) < 10.0,
        }

    def reset(self):
        self.expert_usage = np.zeros(self.num_experts)
        self.total_routing_count = 0


class DynamicMoE(nn.Module):

    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        min_experts=4,
        max_experts=16,
        hidden_dim=256,
        expert_layers=2,
        top_k=2,
        routing_strategy="sparse",
        load_balancing_alpha=0.1,
        config_file=None,
    ):
        super(DynamicMoE, self).__init__()

        if config_file:
            self.config_manager = AIConfigManager(config_file)
            config = self.config_manager.get_config()
            self.input_dim = config.get("input_dim", input_dim)
            self.output_dim = config.get("output_dim", output_dim)
            self.min_experts = config.get("min_experts", min_experts)
            self.max_experts = config.get("max_experts", max_experts)
            self.hidden_dim = config.get("hidden_dim", hidden_dim)
            self.expert_layers = config.get("expert_layers", expert_layers)
            self.top_k = config.get("top_k", top_k)
            routing_strategy = config.get("routing_strategy", routing_strategy)
            load_balancing_alpha = config.get(
                "load_balancing_alpha", load_balancing_alpha
            )
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
            self.experts.append(
                ExpertNetwork(
                    self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers
                )
            )

        self.gating = GatingMechanism(self.input_dim, self.current_experts, self.top_k)

        self.router = RoutingAlgorithm(self.current_experts, routing_strategy)

        self.load_balancer = LoadBalancingLoss(
            self.current_experts, load_balancing_alpha
        )

        self.expert_evaluator = nn.Linear(self.output_dim, 1)

        self.expert_usage = np.zeros(self.max_experts)

    def forward(self, x, return_loss=False):
        batch_size = x.shape[0]

        gate_weights, selected_experts = self.gating(x)

        routed_weights = self.router.route(gate_weights, selected_experts, x)

        output = Tensor(np.zeros((batch_size, self.output_dim)))

        for i, expert in enumerate(self.experts):
            expert_output = expert(x)

            if self.top_k < self.current_experts:
                if i < routed_weights.shape[1]:
                    expert_weight = routed_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor(np.zeros((batch_size, 1)))
            else:
                if i < gate_weights.shape[1]:
                    expert_weight = gate_weights[:, i].unsqueeze(1)
                else:
                    expert_weight = Tensor(np.zeros((batch_size, 1)))

            output = output + expert_output * expert_weight

        self._update_expert_usage(selected_experts, batch_size)

        if return_loss:
            load_balance_loss = self.load_balancer(
                gate_weights, selected_experts, batch_size
            )
            if isinstance(load_balance_loss, Tensor):
                load_balance_loss = load_balance_loss.data
            return output, load_balance_loss

        return output

    def _update_expert_usage(self, selected_experts, batch_size):
        expert_indices = selected_experts.data

        valid_mask = expert_indices < self.current_experts

        np.add.at(self.expert_usage, expert_indices[valid_mask], 1.0)

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

        sorted_indices = sorted(
            range(len(expert_scores)), key=lambda i: expert_scores[i], reverse=True
        )

        if len(self.experts) < self.max_experts and np.std(expert_scores) > 0.1:
            new_expert = ExpertNetwork(
                self.input_dim, self.hidden_dim, self.output_dim, self.expert_layers
            )
            self.experts.append(new_expert)
            self.current_experts += 1

            self.gating = GatingMechanism(
                self.input_dim, self.current_experts, self.top_k
            )
            self.router = RoutingAlgorithm(
                self.current_experts, self.router.routing_strategy
            )
            self.load_balancer = LoadBalancingLoss(
                self.current_experts, self.load_balancer.alpha
            )

            print(f"Added new expert. Current experts: {self.current_experts}")

        elif len(self.experts) > self.min_experts and np.std(expert_scores) < 0.05:
            worst_idx = sorted_indices[-1]
            self.experts.pop(worst_idx)
            self.current_experts -= 1

            self.gating = GatingMechanism(
                self.input_dim, self.current_experts, self.top_k
            )
            self.router = RoutingAlgorithm(
                self.current_experts, self.router.routing_strategy
            )
            self.load_balancer = LoadBalancingLoss(
                self.current_experts, self.load_balancer.alpha
            )

            print(f"Removed worst expert. Current experts: {self.current_experts}")

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "min_experts": self.min_experts,
            "max_experts": self.max_experts,
            "current_experts": self.current_experts,
            "hidden_dim": self.hidden_dim,
            "expert_layers": self.expert_layers,
            "top_k": self.top_k,
            "routing_strategy": self.router.routing_strategy,
            "load_balancing_alpha": self.load_balancer.alpha,
        }

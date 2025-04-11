# Policy Gradient (PG) vs. Proximal Policy Optimization (PPO) 
 
**Comparative Evaluation for Counterfactual Story Rewriting**

## Experimental Design  
Identical setup for both methods to enable fair comparison:  

| Component               | PG Implementation      | PPO Implementation     | Shared Baseline        |
|-------------------------|------------------------|------------------------|------------------------|
| **Base Model**          | Flan-T5                | Flan-T5                | Flan-T5 (MLE-only)     |
| **Pretraining**         | 10 epochs MLE          | 10 epochs MLE          | 10 epochs MLE          |
| **Fine-tuning Epochs**  | 4 epochs PG            | 4 epochs PPO           | N/A                    |
| **Reward Metrics**      | ROUGE-L, BERTScore     | ROUGE-L, BERTScore     | N/A                    |
| **Task**                | Counterfactual ending generation (identical datasets)                    |

---

## Key Algorithmic Differences  

### 1. Policy Gradient (PG)  
**Core Mechanism**:  
```python
# Simplified PG loss (src/pg/models/model.py)
def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards):
    log_probs = self._get_log_probs(generated_tokens, logits)  # Token-level log probabilities
    return -(rewards * log_probs).mean()  # REINFORCE-style update
```
**Characteristics**:  
- Pure on-policy updates  
- High variance (no value baseline)  
- Single update per batch  

### 2. Proximal Policy Optimization (PPO)  
**Core Mechanism**:  
```python
# PPO clipped objective (src/pg/models/model_ppo.py)
def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages):
    ratio = torch.exp(new_log_probs - old_log_probs.detach())  # Importance weight
    clip_epsilon = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
    return -torch.min(surr1, surr2).mean()  # Clipped surrogate objective
```
**Key Additions vs. PG**:  
1. **Value Head** (separate network):  
   ```python
   self.value_head = nn.Sequential(  # Estimates state value
       nn.Linear(d_model, 512),
       nn.ReLU(),
       nn.Linear(512, 1)
   )
   ```
2. **Generalized Advantage Estimation**:  
   ```python
   advantages = rewards + γ * next_values - current_values  # TD residuals
   advantages = gae_λ(advantages)  # Exponential smoothing
   ```
3. **Experience Replay**:  
   ```python
   self.trajectory_buffer = []  # Stores trajectories for multiple updates
   ```

---

## Head-to-Head Comparison  

### Training Dynamics  
| Aspect                | PG                          | PPO                          |
|-----------------------|-----------------------------|------------------------------|
| **Update Frequency**  | 1 update/batch              | 4 updates/batch (configurable) |
| **Gradient Variance** | High (raw rewards)          | Low (value baseline + GAE)    |
| **Memory Usage**      | Low                         | Higher (trajectory buffer)    |
| **Hyperparameters**   | Learning rate only          | clip_ε, γ, λ, entropy_coef    |

### Hypothetical Results (Expected)  
| Metric          | MLE Baseline | MLE + PG   | MLE + PPO  |
|-----------------|-------------|------------|------------|
| **ROUGE-L**     | 0.62        | 0.68 (±0.2)| 0.71 (±0.1)|
| **BERTScore**   | 0.85        | 0.87 (±0.3)| 0.89 (±0.2)|
| **Training Stability** | -    | High variance | Smooth convergence |

### When to Prefer Each Method  
**Choose PG When**:  
- You need simplicity for debugging  
- Training resources are limited  
- Fast experimentation is prioritized  

**Choose PPO When**:  
- Maximizing final reward is critical  
- Training stability matters  
- You can afford slightly higher compute  

---

## Implementation Notes  

### Critical Differences in Code  
1. **Optimizer Setup**:  
   - **PG**: Single optimizer  
     ```python
     torch.optim.AdamW(model.parameters(), lr=5e-5)
     ```
   - **PPO**: Separate value/policy learning rates  
     ```python
     optimizer = AdamW([
         {"params": model.parameters()}, 
         {"params": value_head.parameters(), "lr": 1e-4}
     ], lr=5e-5)
     ```

2. **Training Loop**:  
   - **PG**: Immediate updates  
     ```python
     # Inside training_step():
     loss = pg_loss(batch)
     return loss
     ```
   - **PPO**: Buffered updates  
     ```python
     # Inside training_step():
     self.trajectory_buffer.append(experience)
     if buffer_full:
         for _ in range(ppo_epochs):
             loss = update_ppo()
         return loss
     ```

---

## Conclusion  
This is a **controlled comparison** between two distinct RL approaches applied to the same task. PPO's algorithmic advantages (clipping, value baseline, GAE) *theoretically* should outperform PG, but the actual comparison requires:  

1. **Identical reward functions**  
2. **Same compute budget** (e.g., 4 fine-tuning epochs)  
3. **Identical evaluation metrics**  

The final determination of "which is better" depends on whether PPO's higher sample efficiency and stability outweigh its added complexity for your specific use case.
# Counterfactual Story Rewriting with Policy Gradient Optimization

This project enhances counterfactual story rewriting by combining Maximum Likelihood Estimation (MLE) with Policy Gradient (PG) optimization from Reinforcement Learning (RL). The system uses a T5 transformer model to generate alternative story endings that incorporate specified counterfactual events while maintaining coherence with the original narrative.

## Table of Contents
1. [Implementation Overview](#implementation-overview)
2. [Model Architecture](#model-architecture)
3. [Policy Gradient Implementation](#policy-gradient-implementation)
   - [Reward Calculation](#reward-calculation)
   - [PG Loss Computation](#pg-loss-computation)
4. [Training Workflow](#training-workflow)
5. [Experiments](#experiments)
6. [Configuration](#configuration)
7. [Results and Evaluation](#results-and-evaluation)

---

## Implementation Overview

The system implements a two-phase training approach:
1. **MLE Phase**: Initial supervised training using cross-entropy loss
2. **PG Phase**: Fine-tuning using policy gradient optimization with metric-based rewards

Key components:
- T5 transformer model for sequence generation
- Custom reward function combining multiple metrics
- Policy gradient optimization with baseline subtraction
- Dynamic reward calculation strategies

---

## Model Architecture

The core model is based on Flan-T5 (a variant of T5) with the following customizations:

```python
class FlanT5FineTuner(pl.LightningModule):
    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters('model_name')
        self.model_dir = Path(model_dir)
        self.file_label = file_label
        
        # Initialize T5 model and tokenizer
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Metrics evaluator for reward calculation
        self.metrics_evaluator = MetricsEvaluator()
```

The model handles:
- Sequence generation with sampling
- Log probability calculation
- Reward computation and policy gradient optimization

---

## Policy Gradient Implementation

### Reward Calculation

The system implements several reward calculation strategies configurable via `CONFIG["pg_experiment"]`:

1. **Fixed Baseline**:
   ```python
   rewards = score_pred_edited - CONFIG["baseline_score"]
   ```

2. **Dynamic Baseline** (mean reward):
   ```python
   dynamic_baseline = score_pred_edited.mean().detach()
   rewards = score_pred_edited - dynamic_baseline
   ```

3. **Delta-M1 Strategy** (combines edited and original scores):
   ```python
   delta_m1 = score_pred_edited - score_pred_original
   rewards = score_pred_edited + delta_m1
   dynamic_baseline = rewards.mean().detach()
   rewards = rewards - dynamic_baseline
   ```

4. **Self-Critical Sequence Training (SCST)**:
   ```python
   # Sampled vs greedy generation comparison
   delta_m1_sampled = score_pred_edited - score_pred_original
   delta_m1_greedy = score_pred_edited_greedy - score_pred_original_greedy
   rewards = (score_pred_edited + delta_m1_sampled) - (score_pred_edited_greedy + delta_m1_greedy)
   ```

### PG Loss Computation

The policy gradient loss is calculated as:

```python
def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards, baseline):
    # Stack and normalize logits
    logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
    logits = self.apply_vocab_masking(logits)
    
    # Get log probabilities for generated tokens
    labels_for_indexing = generated_tokens[:, 1:].contiguous()
    token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding tokens
    padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
    token_log_probs = token_log_probs * padding_mask.float()
    
    # Sum log probabilities across sequence
    sequence_log_prob_sum = token_log_probs.sum(dim=1)
    
    # Handle NaN in rewards
    rewards = torch.nan_to_num(rewards, nan=0.0)
    
    # Final loss calculation
    return -(rewards * sequence_log_prob_sum).mean()
```

Key aspects:
1. **Log Probability Calculation**: Uses masked log softmax to get token-level probabilities
2. **Sequence Masking**: Ignores padding tokens in loss calculation
3. **Reward Weighting**: Multiplies sequence log probabilities by their corresponding rewards
4. **Gradient Direction**: Negative sign converts reward maximization to loss minimization

---

## Training Workflow

The training process follows these steps:

1. **Input Preparation**:
   ```python
   input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
   ```

2. **Sequence Generation**:
   ```python
   outputs = self.model.generate(
       input_ids=input_ids,
       attention_mask=attention_mask,
       max_length=CONFIG['max_gen_length'],
       do_sample=True,
       temperature=0.7,
       output_scores=True,
       return_dict_in_generate=True
   )
   ```

3. **Reward Calculation**:
   ```python
   score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
   score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings)
   ```

4. **Loss Computation and Backpropagation**:
   ```python
   pg_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline)
   return pg_loss
   ```

5. **Checkpointing and Early Stopping**:
   ```python
   checkpoint_callback = ModelCheckpoint(
       dirpath=model_dir,
       monitor='validation_pg_loss',
       mode='min',
       save_top_k=1
   )
   
   early_stop_callback = EarlyStopping(
       monitor='validation_pg_loss',
       patience=2,
       mode='min'
   )
   ```

---

## Experiments

The system supports multiple experimental configurations:

1. **MLE-only Training**:
   - 6 epochs of pure supervised learning
   - Baseline for comparison

2. **Combined MLE + PG Training**:
   - Phase 1: 3 epochs MLE
   - Phase 2: 3 epochs PG fine-tuning

3. **Ablation Studies**:
   - Different reward strategies (fixed, dynamic, delta-M1, SCST)
   - Reward component analysis

---

## Configuration

Key configuration parameters (in `CONFIG`):

```python
{
    "model_name": "google/flan-t5-base",
    "pg_experiment": "delta_m1",  # or "fixed", "dynamic", "SCST"
    "learning_rate": 5e-5,
    "batch_size": 8,
    "max_gen_length": 128,
    "baseline_score": 0.5,  # for fixed baseline
    "objective_clipping": True,  # clip negative rewards
    "temperature": 0.7,  # sampling temperature
}
```

---

## Results and Evaluation

Metrics tracked during training:
- `training_pg_loss`: Policy gradient loss
- `training_pg_reward_mean`: Average reward per batch
- `validation_pg_loss`: Validation loss
- Task-specific metrics (ROUGE-L, BERTScore, BARTScore)

Output samples are saved to CSV files for qualitative analysis:

```
validation_details_pg.csv
test_details_pg.csv
```

Each record contains:
- Premise
- Initial event
- Counterfactual event
- Original ending
- Generated ending
- Reward scores

---


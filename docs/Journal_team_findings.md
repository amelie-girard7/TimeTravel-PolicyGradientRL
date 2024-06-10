# Team Notes on Model Training and Dataset Understanding

## Meeting (24/02)
These are the metrics agreed upon for Experiment 4

- Standard metric (either BLEU, ROUGE, BERTScore; any version):

BLEU(generated-text, edited_ending): high is desirable 
BLEU(generated-text, counterfactual): high is desirable
BLEU(generated-text, initial): low is desirable
BLEU(generated-text, original_ending): low is desirable (probably... these are long strings, and may be rather similar also for effective predictions)

We can compute the difference between the desirable scores and the undesrirable scores as a single, overall metric.

To confirm the validity of the above assumptions, we can measure and report the following quantities on the dataset:
BLEU(edited_ending, counterfactual): high is desirable
BLEU(edited_ending, initial): low is desirable
BLEU(edited_ending, original_ending): low is desirable (probably... these are long strings, and may be rather similar)

We can assume that some of these quantities are an "upper bound" for the corresponding scores of the predictions, but I am not sure if this assumption will hold.
For instance, one could assume that BLEU(edited_ending, counterfactual) is always > BLEU(prediction, counterfactual), but it may not be true. Logically, they should be similar.

- BARTScore:
everything said above applies also to BARTScore. We provide the same arguments in the required format, and expect the same trends.

Here 
edited_ending : labels 

## Meeting (16/01)
- The discussion highlights the shift from TensorFlow 1.x (classic use of `tf.Session()`) to TensorFlow 2.x and PyTorch, where the computational graph creation is automatic.
- Emphasis is placed on the dataset from the 2019 paper, noting its value over the need to replicate the paper's experiments (Supervised Learning).

## Dataset and Variable Naming
- Important variables identified in the code corresponding to the dataset fields:
  - `x1`: Premise
  - `x2`: Initial
  - `y`: Original ending (referred to as "s_3:5" in the paper, denoted as uppercase "S")
  - `xx2`: Counterfactual
  - `yy`: Edited ending
- Focus is on understanding these data fields for the supervised experiment, which is crucial.

## Model Insights
- The paper utilizes GPT-2, a decoder-only model, indicating the absence of an encoder. Training involves using input as a "prefix" and the ensuing tokens as the target.
- The loss function prototype and its application as supervised loss are detailed, showing how the edited ending (`yy`) is the actual output target, with other components serving as input.

## Implementation Advice for T5
- For T5 model implementation:
  - Inputs to the encoder should be constructed from `x1x2yx1xx2_ids`.
  - `yy_ids` should be used as the target for the log-likelihood.
- It's strongly recommended to carefully build and verify `x1x2yx1xx2_ids` and `yy_ids` from the JSON file before passing them to the model for training.

## Additional Loss Functions
### Loss-(1): Mask Reconstruction Loss
- `loss_mask_recon = _get_recon_loss(x1x2yx1my_ids, x1x2yx1my_len, x1x2yx1m_len)`
  - Here, the goal is to reconstruct `y` using `x1x2yx1m` as the prefix/input. The `mask_prefix` option defaults to True, excluding `x1x2yx1m` from the NLL calculation.

### Loss-(4): Fine-Tune Loss
- `loss_fine = _get_recon_loss(x1x2y_ids, x1x2y_len, x1x2_len, mask_prefix=False)`
  - This loss aims to reconstruct `x1x2y` using `x1x2` as the prefix/input. Notably, the `mask_prefix` option is set to False, incorporating the prefix in the NLL calculation.

## Key Takeaways
- The importance of the dataset and understanding its fields for experiments.
- Specific advice on handling variables and training with the T5 model, ensuring careful preparation and verification of inputs and targets.
"If it is for the Counterfactul Rewriting research topic and the TimeTravel dataset, I would still use a much smaller model, like T5 or Flan-T5 https://huggingface.co/docs/transformers/model_doc/flan-t5."" Massimo 12/12
- Insight into additional loss functions used in the model's training process.

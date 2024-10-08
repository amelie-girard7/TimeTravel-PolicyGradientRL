# TimeTravel-PolicyGradientRL: Enhancing Counterfactual Story Rewriting via Policy Gradient Optimization

### **Overview**

This project focuses on enhancing counterfactual story rewriting using reinforcement learning and custom reward functions. Counterfactual story rewriting requires models to adjust a narrative's ending based on a hypothetical event, making minimal yet precise changes to maintain narrative coherence. The project introduces a novel training objective and evaluation metrics tailored to this task, leveraging policy gradient methods and reinforcement learning to optimize model performance.

### **Key Features**

- **Custom Training Objective:** A novel training objective designed specifically for counterfactual story rewriting, ensuring minimal, selective, and coherent narrative changes in response to counterfactual events.
  
- **Reinforcement Learning with Custom Reward Functions:** Policy gradient methods are used to directly optimize models with respect to evaluation metrics such as ROUGE, BARTScore, and BERTScore, guiding the model to produce more coherent and contextually appropriate outputs.

- **Counterfactual Rewriting Metrics (CRMs):** New metrics that measure the alignment of the generated endings with the reference while minimizing unnecessary changes from the original story.

### **Repository Structure**

- `src/` - Contains the implementation of the model, training scripts, and evaluation functions.
- `data/` - Example datasets, including the TimeTravel dataset used for counterfactual story rewriting.
- `experiments/` - Scripts to reproduce the experiments and visualize results, including token-level heatmaps for qualitative analysis.
- `README.md` - This README file.

### **Dataset**

We use the [TimeTravel dataset](https://arxiv.org/abs/1911.12399), designed for counterfactual reasoning and narrative understanding tasks. The dataset contains stories with original and edited endings based on counterfactual events.

### **Methodology**

1. **Counterfactual Story Rewriting Task:**
   - **Premise (XP):** The foundational scenario of the story.
   - **Initial Event (XIE):** The original event leading to the story's ending.
   - **Original Ending (XOE):** The original narrative conclusion.
   - **Counterfactual Event (XCE):** A divergent hypothetical event.
   - **Edited Ending (YEE):** The modified conclusion, reflecting the counterfactual event.

2. **Training Objective:**
   The model is optimized to generate the edited ending \(Y_{EE}\), balancing coherence and minimal changes with a custom loss function.

3. **Reinforcement Learning:**
   Policy gradient methods are used to maximize reward functions based on evaluation metrics, encouraging the model to focus on selective and coherent narrative changes.

### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/amelie-girard7/TimeTravel-PolicyGradientRL.git
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### **Running the Model**

To train the model, use the following command:

```bash
python src/train.py --config configs/default.yaml
```

For evaluation:

```bash
python src/evaluate.py --model-checkpoint path/to/checkpoint
```

### **Evaluation**

The model is evaluated using the following metrics:

- **ROUGE:** Measures the overlap between generated and reference text.
- **BARTScore:** A scoring method for text generation quality based on BART.
- **BERTScore:** Measures similarity between the generated and reference endings using BERT embeddings.
- **CRMs:** Custom metrics to evaluate counterfactual alignment and minimal narrative changes.

### **Results**

Our experiments demonstrate that the proposed methods outperform baseline models, effectively incorporating counterfactual reasoning into the generated stories while maintaining coherence and minimizing unnecessary changes.

### **Citations**

If you use this work, please cite the following paper:

```
@article{girard2024counterfactual,
  title={Enhancing Counterfactual Story Rewriting via Policy Gradient Optimization and Custom Rewards},
  author={Girard, Amelie and Jauregi Unanue, Inigo and Piccardi, Massimo},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


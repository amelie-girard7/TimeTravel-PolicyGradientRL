# Introduction

The primary aim of this study is to measure the T5 model's efficacy in a story rewriting task, focusing on performance measurement through established evaluation benchmarks such as Sacre BLEU, Sacre ROUGE, and context-aware metrics like BERT and BART. This evaluation utilises a dataset comprising stories with modified conclusions (edited_endings) based on a counterfactual(hypothetical) scenario. The objective is to gauge the model's ability to modify the original story ending to reflect the counterfactual scenario within a supervised learning framework.

## Dataset
- **train_supervised_small.json**: Contains 16,752 examples featuring human-modified endings.
- **dev_data.json**: Comprises 1,871 examples with human-modified endings.
- **test_data.json**: Includes 1,871 examples with human-modified endings.

Dataset Structure:
- **Premise**: Initial context of the story (e.g., "Andrea wanted a picture of her jumping.").
- **Initial**: Event leading up to the counterfactual alteration (e.g., "She set the camera up.").
- **Counterfactual**: Hypothetical scenario introduced (e.g., "She asked her friend to draw one.").
- **Original Ending**: The story's conclusion without the counterfactual modification (e.g., "Then, she jumped in the air...").
- **Edited Ending**: The altered conclusion by the model accounting for the counterfactual scenario (e.g., "Then, she jumped in the air to demonstrate...").

#### Example

```json
{
  "story_id": "4fd7d150-b080-4fb1-a592-8c27fa6e1fc8",
  "premise": "Andrea wanted a picture of her jumping.",
  "initial": "She set the camera up.",
  "counterfactual": "She asked her friend to draw one.",
  "original_ending": "Then, she jumped in the air. The picture kept coming out wrong. It took twenty tries to get it right.",
  "edited_ending": [
    "Then, she jumped in the air to demonstrate how she wanted it to look.",
    "The picture kept coming out wrong.",
    "It took drawing it several times to get it right."
  ]
}
```

# Experiment Environment:

**Flan-T5**
For model documentation: https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5

| Model                  | Parameters             | Download Size |
|------------------------|------------------------|---------------|
| google/flan-t5-base    | Roughly 247M           |               |
| google/flan-t5-large   | About 770M             | 1GB           |     

**Metrics**
- BERT Score Documentation: https://github.com/Tiiiger/bert_score (Model: microsoft/deberta-xlarge-mnli)
- BART Score Documentation: https://github.com/neulab/BARTScore (Model: facebook/bart-large-cnn)
- ROUGE Metric Info: https://github.com/pltrdy/rouge
- BLEU Score Info: https://github.com/mjpost/sacrebleu

| Metric      | Range    | Desired Outcome |
|-------------|----------|-----------------|
| Sacre-Bleu  | [0-100]  | High            |
| Sacre-Rouge | [0-1]    | High            |
| BART        | [-Inf, 0]| Higher (less negative) |
| BERT        | [0-1]    | High            |

# Expected outcome

To create a comprehensive table including the column for interpretation as you requested, let's compile the expectations and interpretations for each similarity comparison between different elements of the text (such as prediction, edited_ending, counterfactual, initial, original_ending, and premise). This table will help clarify the expectations from the model's outputs in relation to the provided inputs and the edited endings.


|  Similarity                      | Expected Outcome |Interpretation  
|----------------------------------|------------------|---------------------------------------------------------------------------------------|
| (prediction, edited_ending)      |High              | We expect a high similarity as the model's predictions should closely match the edited endings, indicating accurate reflection of the counterfactual in the rewrite. |
| (prediction, counterfactual)     | High             | High similarity is expected since the model's predictions should incorporate elements or concepts from the counterfactual scenario.                                 |
| (prediction, initial)            | Low              | Low similarity is desirable, indicating that the model moves beyond the initial setup to explore new narrative directions.                                         |
| (prediction, original_ending)    | Low (probably)   | A low similarity suggests that the model effectively generates new endings distinct from the original, although some overlap may occur due to narrative constraints.|
| (prediction, Premise)            | High             | High similarity is expected because both the model's predictions and the premise should share thematic or narrative consistency.                                    |
| (edited_ending, counterfactual)  | High             | High similarity is expected as the edited endings are designed to reflect the introduced counterfactual scenario directly.                                          |
| (edited_ending, initial)         | Low              | Low similarity is desirable, indicating that the edited endings diverge significantly from the initial events to accommodate the counterfactual.                    |
| (edited_ending, original_ending) | Low              | A low similarity indicates successful diversion from the original ending due to the integration of the counterfactual, despite potential narrative overlap.         |


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

#### Original paper Results 

| Model       | BLEU-4 | ROUGE-L | BERT |
|-------------|--------|---------|------|
| GPT + Sup   | 80.09  | 75.03   | 64.15|
| GPT2-S + Sup| 79.03  | 73.31   | 64.14|
| GPT2-M + Sup| 76.63  | 74.42   | 64.06|


## Experiments 26/03 

### Mask the original ending

((model_2024-03-26-14), (model_2024-03-26-16))
In order to evaluate the model's capability to generate an edited ending without being directly influenced by the original ending, we modified the input sequence construction within the `preprocess_data` function. Essentially,we removed or masked the portion of the input that includes the original ending when constructing `input_sequence`. 

```python
def preprocess_data(row, tokenizer, CONFIG):
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']} {separator_token} "  # Mask the original ending by not including it here.
            f"{row['premise']} {row['counterfactual']}"
        )
```
This adjustment ensures that the model's input sequence does not include the original ending, thereby "masking" it during the model's attempt to generate an edited ending based on the premise, initial event, and counterfactual scenario. This approach helps in assessing the model's ability to creatively alter the ending without leaning on the original ending's content.

### Data augmentation

## Experiment 22/03

- Epocs: 6 instead of 3 
- Learning rate 2e-5 instead of 0.001
(Other considerations - code optimisation)



## Experiment 4 



Note: I've added Premise for comparability with the paper



### Experiment 4 - 

12/03: train_supervised_small.json: 16,752 examples with edited ending.( Similarity [-Inf-0])
14/03 train_supervised_large.json: 28,363 examples with edited ending.(Large similarity)

| BART                                 | base+small[-Inf-0]| base+large data    | T5-large+data-small |
|--------------------------------------|-------------------|--------------------|---------------------|
|bart_(edited_ending, counterfactual)  |-3.270855665206909 | -3.270855665206909 |-3.270648241043091   |
|bart_(edited_ending, initial)         |-3.405696153640747 | -3.405696153640747 |-3.400423765182495   |
|bart_(edited_ending, original_ending) |-1.4418021440505981| -1.4418021440505981|-1.4388474225997925  |
|bart_(prediction, counterfactual)     |-3.444650411605835 | -3.4401299953460693|-3.413386821746826   |
|bart_(prediction, edited_ending)      |-1.7036479711532593| -1.7080293893814087|-1.6894985437393188  |
|bart_(prediction, initial)            |-3.3090765476226807| -3.317261219024658 |-3.318594217300415   |
|bart_(prediction, original_ending)    |-0.3739939033985138| -0.4007585644721985|-0.4233478009700775  |



| BERT metric                          | base+small [0-1]   | base+large data   | T5-large+data-small|
|--------------------------------------|--------------------|-------------------|--------------------|
| bert_edited_ending_cf_f1             | 0.5964024662971497 |0.5964024662971497 |0.5962095856666565  |
| bert_edited_ending_cf_precision      | 0.5502906441688538 |0.5502906441688538 |0.5502889752388     |
| bert_edited_ending_cf_recall         | 0.653448224067688  |0.653448224067688  |0.6530463099479675  |
| bert_edited_ending_initial_f1        | 0.5916143655776978 |0.5916143655776978 |0.5914896130561829  |
| bert_edited_ending_initial_precision | 0.5412111282348633 |0.5412111282348633 |0.5412472486495972  |
| bert_edited_ending_initial_recall    | 0.6547386646270752 |0.6547386646270752 |0.6544120907783508  |
| bert_edited_ending_original_f1       | 0.848354697227478  |0.848354697227478  |0.8488994836807251  |
| bert_edited_ending_original_precision| 0.8407038450241089 |0.8407038450241089 |0.8412861824035645  |
| bert_edited_ending_original_recall   | 0.8570297360420227 |0.8570297360420227 |0.8575602769851685  |
| bert_prediction_cf_f1                | 0.5924401879310608 |0.5922539830207825 |0.5937129259109497  |
| bert_prediction_cf_precision         | 0.5504221320152283 |0.5502236485481262 |0.551712691783905   |
| bert_prediction_cf_recall            | 0.6437558531761169 |0.6436275839805603 |0.6450079083442688  |
| bert_prediction_edited_f1            | 0.8435136675834656 |0.8426817655563354 |0.8442564010620117  |
| bert_prediction_edited_precision     | 0.8520939946174622 |0.85141921043396   |0.8529777526855469  |
| bert_prediction_edited_recall        | 0.8359667062759399 |0.8349965810775757 |0.8365933895111084  |
| bert_prediction_initial_f1           | 0.6001779437065125 |0.5996060371398926 |0.5997409820556641  |
| bert_prediction_initial_precision    | 0.5512555837631226 |0.550765872001648  |0.5513353943824768  |
| bert_prediction_initial_recall       | 0.6608282327651978 |0.6601568460464478 |0.6596580743789673  |
| bert_prediction_original_f1          | 0.9875152707099915 |0.9851989150047302 |0.983555257320404   |
| bert_prediction_original_precision   | 0.9875959157943726 |0.9856005311012268 |0.983866810798645   |
| bert_prediction_original_recall      | 0.9874710440635681 |0.9848557114601135 |0.983290433883667   |


| Bleu metric                          | base+small [0-100] |base+large data    |T5-large+data-small |
|--------------------------------------|--------------------|-------------------|--------------------|
| bleu_edited_ending_cf                | 0.18647107481956482|0.18647107481956482|0.20867815613746643 |
| bleu_edited_ending_initial           | 0.1779923439025879 |0.1779923439025879 |0.1931639164686203  |
| bleu_edited_ending_original          | 0.06398702412843704|0.06398702412843704|0.06363406777381897 |
| bleu_prediction_cf                   | 15.545328140258789 |15.85953426361084  |15.051192283630371  |
| bleu_prediction_edited               | 78.89669799804688  |82.37779235839844  |81.10931396484375   |
| bleu_prediction_initial              | 11.120792388916016 |11.24506664276123  |10.887622833251953  |
| bleu_prediction_original             | 93.21022033691406  |95.76140594482422  |100.0               |



| Rouge metric                         | base+small [0-1]    |base+large data     |T5-large+data-small |
|--------------------------------------|---------------------|--------------------|--------------------|
| rouge_edited_ending_cf_rouge-1_f     | 0.17747779190540314 |0.17747779190540314 |0.17699365317821503 |
| rouge_edited_ending_cf_rouge-1_p     | 0.1295394003391266  |0.1295394003391266  |0.12940001487731934 |
| rouge_edited_ending_cf_rouge-1_r     | 0.30741605162620544 |0.30741605162620544 |0.30621516704559326 |
| rouge_edited_ending_cf_rouge-2_f     | 0.024872800335288048|0.024872800335288048|0.02463381178677082 |
| rouge_edited_ending_cf_rouge-2_p     | 0.017150932922959328|0.017150932922959328|0.017023704946041107|
| rouge_edited_ending_cf_rouge-2_r     | 0.0506436750292778  |0.0506436750292778  |0.050234466791152954|
| rouge_edited_ending_cf_rouge-l_f     | 0.16370658576488495 |0.16370658576488495 |0.1630040854215622  |
| rouge_edited_ending_cf_rouge-l_p     | 0.11930356174707413 |0.11930356174707413 |0.11901543289422989 |
| rouge_edited_ending_cf_rouge-l_r     | 0.28467464447021484 |0.28467464447021484 |0.28306844830513    |

|rouge_edited_ending_initial_rouge-1_f | 0.15380537509918213 |0.15380537509918213 |0.15383368730545044 |
|rouge_edited_ending_initial_rouge-1_p | 0.10816317051649094 |0.10816317051649094 |0.10822901129722595 |
|rouge_edited_ending_initial_rouge-1_r | 0.28547996282577515 |0.28547996282577515 |0.2853497862815857  |  
|rouge_edited_ending_initial_rouge-2_f | 0.01875416561961174 |0.01875416561961174 |0.019036682322621346|
|rouge_edited_ending_initial_rouge-2_p | 0.012573624029755592|0.012573624029755592|0.012737231329083443|
|rouge_edited_ending_initial_rouge-2_r | 0.04027299955487251 |0.04027299955487251 |0.04126160591840744 | 
|rouge_edited_ending_initial_rouge-l_f | 0.14384855329990387 |0.14384855329990387 |0.1435905545949936  |
|rouge_edited_ending_initial_rouge-l_p | 0.10100927948951721 |0.10100927948951721 |0.10084190964698792 |
|rouge_edited_ending_initial_rouge-l_r | 0.2681836485862732  |0.2681836485862732  |0.26763278245925903 |

| rouge_edited_ending_original_rouge-1_f| 0.7531278133392334 |0.7531278133392334  |0.753814697265625   |
| rouge_edited_ending_original_rouge-1_p| 0.7389482855796814 |0.7389482855796814  |0.7405248284339905  |
| rouge_edited_ending_original_rouge-1_r| 0.7753328680992126 |0.7753328680992126  |0.775071382522583   |
| rouge_edited_ending_original_rouge-2_f| 0.6301290988922119 |0.6301290988922119  |0.630500316619873   |
| rouge_edited_ending_original_rouge-2_p| 0.6145238280296326 |0.6145238280296326  |0.6156041622161865  |
| rouge_edited_ending_original_rouge-2_r| 0.6531437635421753 |0.6531437635421753  |0.6527891755104065  |
| rouge_edited_ending_original_rouge-l_f| 0.7497166395187378 |0.7497166395187378  |0.750571608543396   |
| rouge_edited_ending_original_rouge-l_p| 0.7355568408966064 |0.7355568408966064  |0.73729407787323    |
| rouge_edited_ending_original_rouge-l_r| 0.7718437910079956 |0.7718437910079956  |0.771765947341919   |

| rouge_prediction_cf_rouge-1_f         |0.16842211782932281 |0.1695558726787567  |0.17112833261489868 |
| rouge_prediction_cf_rouge-1_p         |0.12465791404247284 |0.12571480870246887 |0.12683571875095367 |
| rouge_prediction_cf_rouge-1_r         |0.28206369280815125 |0.2829330563545227  |0.2857707440853119  |
| rouge_prediction_cf_rouge-2_f         |0.022407902404665947|0.022306649014353752|0.02265637367963791 |
| rouge_prediction_cf_rouge-2_p         |0.015689916908740997|0.015656059607863426|0.015895504504442215|
| rouge_prediction_cf_rouge-2_r         |0.04370618984103203 |0.043212100863456726|0.04397522285580635 |
| rouge_prediction_cf_rouge-l_f         |0.15738092362880707 |0.15837831795215607 |0.1590966135263443  |
| rouge_prediction_cf_rouge-l_p         |0.1162935346364975  |0.1172562763094902  |0.11771118640899658 |
| rouge_prediction_cf_rouge-l_r         |0.26453089714050293 |0.26514536142349243 |0.26676446199417114 |

| rouge_prediction_edited_rouge-1_f     | 0.7477840781211853 |0.7468565106391907  |0.7487048506736755  |
| rouge_prediction_edited_rouge-1_p     | 0.7710343599319458 |0.7714491486549377  |0.7718952894210815  |
| rouge_prediction_edited_rouge-1_r     | 0.7327389717102051 |0.7309657335281372  |0.7338786125183105  |
| rouge_prediction_edited_rouge-2_f     | 0.6210501194000244 |0.6202576160430908  |0.6224623918533325  |
| rouge_prediction_edited_rouge-2_p     | 0.6443043351173401 |0.6447920203208923  |0.6452279090881348  |
| rouge_prediction_edited_rouge-2_r     | 0.6052320599555969 |0.603620171546936   |0.607215940952301   | 
| rouge_prediction_edited_rouge-l_f     | 0.7444990277290344 |0.7435429692268372  |0.745520293712616   |
| rouge_prediction_edited_rouge-l_p     | 0.7676512598991394 |0.7680289149284363  |0.7686030268669128  |
| rouge_prediction_edited_rouge-l_r     | 0.7294994592666626 |0.7277039289474487  |0.7307514548301697  |

| rouge_prediction_initial_rouge-1_f    |0.15811428427696228 |0.15806500613689423 |0.1568852961063385  |
| rouge_prediction_initial_rouge-1_p    |0.11256470531225204 |0.11266595125198364 |0.11189664155244827 |
| rouge_prediction_initial_rouge-1_r    |0.2829751670360565  |0.2821216881275177  |0.27971887588500977 |
| rouge_prediction_initial_rouge-2_f    |0.01917721889913082 |0.018890324980020523|0.018419884145259857|
| rouge_prediction_initial_rouge-2_p    |0.013106497004628181|0.012934474274516106|0.01259550265967846 |
| rouge_prediction_initial_rouge-2_r    |0.03867366537451744 |0.037802111357450485|0.03708050772547722 |
| rouge_prediction_initial_rouge-l_f    |0.14854687452316284 |0.14885403215885162 |0.147553950548172   |
| rouge_prediction_initial_rouge-l_p    |0.1055985763669014  |0.10597000271081924 |0.10508032888174057 |
| rouge_prediction_initial_rouge-l_r    |0.2669052481651306  |0.2665397822856903  |0.2640511691570282  |

| rouge_prediction_original_rouge-1_f   |0.9856066107749939  |0.981619656085968   |0.9784668684005737  |
| rouge_prediction_original_rouge-1_p   |0.987214207649231   |0.9849103093147278  |0.9810206294059753  |
| rouge_prediction_original_rouge-1_r   |0.9842634201049805  |0.9790352582931519  |0.9764140248298645  |
| rouge_prediction_original_rouge-2_f   |0.9729441404342651  |0.9676569700241089  |0.9598492980003357  |
| rouge_prediction_original_rouge-2_p   |0.9739639759063721  |0.9706407785415649  |0.9611953496932983  |
| rouge_prediction_original_rouge-2_r   |0.9721643924713135  |0.9653595685958862  |0.9589384198188782  |
| rouge_prediction_original_rouge-l_f   |0.9856066107749939  |0.9816002249717712  |0.9784042239189148  |
| rouge_prediction_original_rouge-l_p   |0.987214207649231   |0.9848905205726624  |0.980957567691803   |
| rouge_prediction_original_rouge-l_r   |0.9842634201049805  |0.9790161848068237  |0.9763516187667847  |

|              val_loss                 |1.0465151071548462  |1.0366238355636597  |0.9303057193756104  |










#### Experiment 4.1 Bleu score: (03/03)

Interpreting the SacreBLEU scores requires understanding how BLEU scores function in the context of natural language processing. BLEU (Bilingual Evaluation Understudy) scores compare machine-generated text to reference texts, assessing the quality of the generated text. The scores range from 0 to 100, where higher scores indicate greater similarity between the generated text and the reference text, suggesting better quality generation.

The SacreBLEU metric extends the original BLEU by providing a standardized way to calculate BLEU scores, ensuring consistent and comparable scores across different evaluations. It does this by taking the generated text (hypotheses) and comparing it against one or more reference texts. These comparisons are done on the corpus level, meaning SacreBLEU considers the entire set of generated texts and their corresponding reference texts as a whole, rather than evaluating each pair in isolation. This approach helps mitigate some of the variability and potential biases that can arise from sentence-level evaluations.

Below are the results of experiment:

| Metric                           | Score               | Desired outcome         | Interpretation                                                                                     |
|----------------------------------|---------------------|---------------------------------|----------------------------------------------------------------------------------------------------|
| BLEU(generated-text, edited_ending)  | 81.499              | High               | Indicates a high degree of similarity between the generated text and the labels, suggesting effective learning and generation capabilities.            |
| BLEU(generated-text, counterfactual) | 16.467              | High               | A lower score here might be expected, as the counterfactual introduces a hypothetical change not necessarily reflected in the text directly. This score suggests room for improvement in integrating counterfactual nuances. |
| BLEU(generated-text, initial)        | 16.617              | Low                | The model does not overly rely on the initial event in generating the ending, which is desirable as it shows adaptability to the counterfactual change.  |
| BLEU(generated-text, original_ending)| 95.262              | Low  (probably)     | A high score indicates the generated text is very similar to the original ending, which might suggest a lack of sufficient adaptation to the counterfactual change. However, considering these are long strings and may be inherently similar for effective predictions, this might not be entirely negative. |

Based on your criteria, let's also consider the BLEU scores between the edited endings and other story components as a way to establish a baseline or "upper bound" for the corresponding scores of the predictions:

| Metric                                | Score               | Interpretation Criteria         | Interpretation                                                                                     |
|---------------------------------------|---------------------|---------------------------------|----------------------------------------------------------------------------------------------------|
| BLEU(edited_ending, counterfactual)   | 13.745              | High          | Indicates that the labels share some similarity with the counterfactual scenarios but also suggests room for nuanced interpretation and generation by the model. |
| BLEU(edited_ending, initial)          | 10.881              | Low               | Aligns with the expectation that the edited endings should diverge from the initial event, suggesting the model should focus on integrating the counterfactual change rather than mimicking the initial scenario. |
| BLEU(edited_ending, original_ending)  | 83.130              | Low (probably)     | A high score suggests that the edited endings are quite similar to the original endings, providing a context where the model might inherently score high when comparing generated text to the original ending due to the similarity in structure and content between the edited and original endings. |

Note: ( to be discussed)
BLEU(edited_ending, original_ending) : I don't agree with this interpretation criteria, in the paper High is desirable. Minimum changes to the story ending based on the counterfactual was desired.

The model shows strong alignment with the edited endings, indicating effective learning and prediction capabilities. The lower scores with the counterfactual and initial scenarios suggest that while the model adapts to the counterfactual changes, there's potential for further refinement to better integrate these nuances into the generated text. The high similarity between generated texts and the original ending suggests a need to ensure the model sufficiently diverges in response to counterfactual changes, although this interpretation is nuanced by the inherent similarity between edited and original endings in your dataset.

#### 4.1 Bleu Rouge score: (04/03)

| Metric Type | Comparison                         | Score                  | Metric Name                          |
|-------------|------------------------------------|------------------------|--------------------------------------|
| ROUGE-1     | Edited Ending vs. Original         | 0.7531278133392334     | rouge_edited_ending_original_rouge-1_f |
| ROUGE-1     | Edited Ending vs. Counterfactual   | 0.17747779190540314    | rouge_edited_ending_cf_rouge-1_f    |
| ROUGE-1     | Edited Ending vs. Initial          | 0.15380537509918213    | rouge_edited_ending_initial_rouge-1_f |
| ROUGE-1     | Prediction vs. Original            | 0.7677057385444641     | rouge_prediction_original_rouge-1_f |
| ROUGE-1     | Prediction vs. Edited              | 0.5818418860435486     | rouge_prediction_edited_rouge-1_f   |
| ROUGE-1     | Prediction vs. Counterfactual      | 0.16079357266426086    | rouge_prediction_cf_rouge-1_f       |
| ROUGE-1     | Prediction vs. Initial             | 0.1546783745288849     | rouge_prediction_initial_rouge-1_f  |
| ROUGE-2     | Edited Ending vs. Original         | 0.6301290988922119     | rouge_edited_ending_original_rouge-2_f |
| ROUGE-2     | Edited Ending vs. Counterfactual   | 0.024872800335288048   | rouge_edited_ending_cf_rouge-2_f    |
| ROUGE-2     | Edited Ending vs. Initial          | 0.01875416561961174    | rouge_edited_ending_initial_rouge-2_f |
| ROUGE-2     | Prediction vs. Original            | 0.7173408269882202     | rouge_prediction_original_rouge-2_f |
| ROUGE-2     | Prediction vs. Edited              | 0.44451233744621277    | rouge_prediction_edited_rouge-2_f   |
| ROUGE-2     | Prediction vs. Counterfactual      | 0.01878502033650875    | rouge_prediction_cf_rouge-2_f       |
| ROUGE-2     | Prediction vs. Initial             | 0.01721237786114216    | rouge_prediction_initial_rouge-2_f  |
| BLEU        | Edited Ending vs. Counterfactual   | 0.18647107481956482    | bleu_edited_ending_cf               |
| BLEU        | Edited Ending vs. Initial          | 0.1779923439025879     | bleu_edited_ending_initial          |
| BLEU        | Edited Ending vs. Original         | 0.06398702412843704    | bleu_edited_ending_original         |
| BLEU        | Prediction vs. Counterfactual      | 16.467029571533203     | bleu_prediction_cf                  |
| BLEU        | Prediction vs. Edited              | 81.49915313720703      | bleu_prediction_edited              |
| BLEU        | Prediction vs. Initial             | 16.617429733276367     | bleu_prediction_initial             |
| BLEU        | Prediction vs. Original            | 95.26201629638672      | bleu_prediction_original            |




#### 4.3 BERT score: (04/03)

| Metric Type | Comparison                          | Score                | Metric Name                    |
|-------------|-------------------------------------|----------------------|--------------------------------|
| BERT        | Edited Ending vs. Counterfactual    | 0.5746781229972839   | bert_edited_ending_cf          |
| BERT        | Edited Ending vs. Initial           | 0.5564911365509033   | bert_edited_ending_initial     |
| BERT        | Edited Ending vs. Original          | 0.8539996147155762   | bert_edited_ending_original    |
| BERT        | Prediction vs. Counterfactual       | 0.5689771175384521   | bert_prediction_cf             |
| BERT        | Prediction vs. Edited               | 0.7977718710899353   | bert_prediction_edited         |
| BERT        | Prediction vs. Initial              | 0.5713935494422913   | bert_prediction_initial        |
| BERT        | Prediction vs. Original             | 0.8426278233528137   | bert_prediction_original       |
| N/A         | Model Validation Loss               | 1.044119119644165    | val_loss                       |


BERT (Bidirectional Encoder Representations from Transformers) operates fundamentally differently from models that are purely used for generating text or embeddings for similarity comparisons. Here's a brief overview of how BERT works:

 For similarity comparisons, BERT generates embeddings for texts, where each embedding captures semantic information of the text. Similarity between texts can be measured by computing the cosine similarity between their embeddings, with higher scores indicating greater similarity.

BERT's ability to understand the context and nuances of language makes it highly effective for tasks involving natural language understanding, including generating embeddings for similarity comparisons as demonstrated in the table above.

#### Interpretation 
Understood, let's enhance the interpretations to focus more on the implications of the scores within their context, without altering the scores and avoiding repetition of the comparison details already provided in the table.

### Enhanced Interpretations

| Metric Type | Comparison                         | Score                  | Range  | Desired Outcome | Interpretation                                                                                                                      |
|-------------|------------------------------------|------------------------|--------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| ROUGE-1     | Edited Ending vs. Original         | 0.7531278133392334     | 0-1    | Low             | A high score suggests significant content overlap, indicating less narrative diversity than desired.                                 |
| ROUGE-1     | Edited Ending vs. Counterfactual   | 0.17747779190540314    | 0-1    | High            | A low score here points to minimal direct word reuse, aligning with the goal of integrating new counterfactual elements effectively. |
| ROUGE-1     | Edited Ending vs. Initial          | 0.15380537509918213    | 0-1    | Low             | This low score is favorable, reflecting substantial narrative alteration from the story's beginning.                                |
| ROUGE-1     | Prediction vs. Original            | 0.7677057385444641     | 0-1    | Low             | The closeness to the original's wording suggests a need for increased novelty in model outputs.                                      |
| ROUGE-1     | Prediction vs. Edited              | 0.5818418860435486     | 0-1    | High            | Indicates a moderate alignment with human edits, suggesting room for improvement in mimicking desired changes.                       |
| ROUGE-1     | Prediction vs. Counterfactual      | 0.16079357266426086    | 0-1    | High            | Reflects the model's challenge in fully capturing counterfactual nuances, given the score's proximity to a low overlap.              |
| ROUGE-1     | Prediction vs. Initial             | 0.1546783745288849     | 0-1    | Low             | Demonstrates the model's effectiveness in diverging from the initial scenario, as evidenced by low overlap.                         |
| ROUGE-2     | Edited Ending vs. Original         | 0.6301290988922119     | 0-1    | Low             | A high bigram similarity score here signals potential over-reliance on the original narrative structure.                             |
| ROUGE-2     | Edited Ending vs. Counterfactual   | 0.024872800335288048   | 0-1    | High            | Extremely low bigram overlap underscores the model's capacity to innovate beyond straightforward narrative extensions.             |
| ROUGE-2     | Edited Ending vs. Initial          | 0.01875416561961174    | 0-1    | Low             | The negligible bigram similarity highlights the model's success in significantly transforming the story from its start.             |
| BLEU        | Edited Ending vs. Counterfactual   | 0.18647107481956482    | 0-100  | High            | The score indicates a nuanced but limited precision in adapting to the counterfactual, suggesting areas for improvement.            |
| BLEU        | Edited Ending vs. Initial          | 0.1779923439025879     | 0-100  | Low             | Reflects the model's ability to evolve the narrative from the initial scenario, though the proximity hints at conservative changes. |
| BLEU        | Edited Ending vs. Original         | 0.06398702412843704    | 0-100  | Low             | Demonstrates significant deviation from the original, with the low precision score indicating substantial narrative innovation.     |
| BLEU        | Prediction vs. Counterfactual      | 16.467029571533203     | 0-100  | High            | The relatively low score for a BLEU metric suggests the model's limited effectiveness in counterfactual adaptation.                 |
| BLEU        | Prediction vs. Edited              | 81.49915313720703      | 0-100  | High            | High precision with edited endings shows the model's proficiency in capturing the intended narrative adjustments.                   |
| BLEU        | Prediction vs. Initial             | 16.617429733276367     | 0-100  | Low             | Indicates the model's creative departure from the initial setup, but suggests further room for diversification.                     |
| BLEU        | Prediction vs. Original            | 95.26201629638672      | 0-100  | Low             | The high precision score reveals the model's tendency to replicate original narratives closely, highlighting a need for more novelty.|

## Experiment 1: Initial Prototype
(Wed 14/02/2023 - branch prototype-end_end). 

### Results Table and Interpretation

| Metric               | FLAN-T5 Test Results | Paper Benchmark | Interpretation                                         |
|----------------------|----------------------|---------------------------|--------------------------------------------------------|
| avg_bleu             | 0.533                | -                         | Moderate accuracy in text generation compared to targets. |
| rouge1_avg_fmeasure  | 0.754                | -                         | Good balance between precision and recall in capturing key information. |
| rouge2_avg_fmeasure  | 0.633                | -                         | Demonstrates the model's ability to replicate more complex sequences. |
| rougeL_avg_fmeasure  | 0.740                | -                         | High effectiveness in capturing longer sequences faithfully. |
| test_loss            | 1.042                | -                         | Indicates potential for improvement in prediction accuracy. |


- **BLEU Score (avg_bleu):** 0.533. This score measures the model's accuracy in reproducing the exact sequences of words in the target texts. A score closer to 1 indicates higher accuracy. In the context of this model, a BLEU score of approximately 0.533 suggests a moderate level of accuracy in generating text that matches the reference sequences.

- **ROUGE Scores:** These scores evaluate the overlap between the generated text and the reference texts across several dimensions:
  - **F-measure:** Reflects the balance between precision (exactness) and recall (completeness) in matching the reference texts. The model achieves over 0.74 in ROUGE-L, indicating a good balance in capturing the essence of the target texts.
  - **Precision:** Indicates the proportion of correctly generated words against all generated words. With values above 0.75 for ROUGE-1, the model demonstrates a high level of precision.
  - **Recall:** Measures the proportion of correctly generated words against all words in the reference texts. The model shows strong recall rates, suggesting it can capture a significant portion of the reference content.

- **Test Loss:** 1.042. This value indicates the model's average error in predicting the target sequences during testing. Lower loss values denote better performance. ??The observed loss suggests the model has room for improvement but is on a promising trajectory??.

### Detailed Examples and Interpretation

Let's break down specific examples to illustrate how the model performs:

1. **Generated Text:** "He felt nervous getting in the water for the first time. Eventually, he got the hang of propelling himself in the water. Wendell became a great swimmer."
   
   **Decoded Target:** "He always felt nervous getting into the water, before his accident. Eventually, he probably would have got the hang of it. Wendell could have became a great swimmer, if it wasn't for that sad day in the pond."

   **Detailed Interpretation:** The model's output closely follows the narrative arc of the reference but optimistically concludes Wendell becoming a great swimmer, unlike the target text which hints at a tragic turn. This demonstrates the model's capability in narrative continuation but also highlights its challenge in capturing underlying tones or implied narratives.

2. **Generated Text:** "But they were from two different social backgrounds. They tried and tried to make their love work. But it just wasn't meant to be."
   
   **Decoded Target:** "They were from two different social backgrounds. They tried and tried to make their love work. But it just wasn't meant to be."

   **Detailed Interpretation:** Here, the model nearly perfectly replicates the target narrative, indicating a strong alignment in simpler, more direct narratives. The slight addition of "But" at the beginning of the generated text introduces a negligible deviation, showcasing the model's precision in simpler contexts.


The FLAN-T5 model demonstrates a commendable ability to generate text that aligns with the given context and targets, as evidenced by its BLEU and ROUGE scores. While there is room for optimization, particularly in enhancing the model's precision and reducing test loss, its current performance showcases a robust foundation for narrative generation tasks. Comparing these results with benchmarks from the literature can further elucidate the model's positioning within the field and guide future improvements.

## Experiment 2: Optimized Architecture
(Mon 19/02/2024 - branch prototype-end_to_end)

We have optimised the architecture to handel the tokenisation and padding at the level of the preprocess and collate_fn functions.


### Results Table and Interpretation

| Metric                   | Score                 |
|--------------------------|-----------------------|
| Average BLEU Score       | 1.018                 |
| ROUGE-1 F-measure        | 2.279                 |
| ROUGE-1 Precision        | 3.172                 |
| ROUGE-1 Recall           | 1.812                 |
| ROUGE-2 F-measure        | 1.813                 |
| ROUGE-2 Precision        | 2.580                 |
| ROUGE-2 Recall           | 1.425                 |
| ROUGE-L F-measure        | 2.201                 |
| ROUGE-L Precision        | 3.063                 |
| ROUGE-L Recall           | 1.750                 |
| ROUGE-Lsum F-measure     | 2.201                 |
| ROUGE-Lsum Precision     | 3.063                 |
| ROUGE-Lsum Recall        | 1.750                 |
| Validation Loss          | 1.026                 |


- **Average BLEU Score:** 1.018, indicating a good level of accuracy in matching the reference texts. This score suggests that the model can reproduce specific sequences of words with a reasonable degree of accuracy.

- **ROUGE Metrics:**
  - **ROUGE-1 F-measure:** 2.279, showing the model's capability in capturing individual words from the reference texts.
  - **ROUGE-2 F-measure:** 1.813, indicating the model's effectiveness in capturing two-word phrases, a bit lower than single-word accuracy but still commendable.
  - **ROUGE-L F-measure:** 2.201, reflecting the model's ability to generate longer, coherent sequences similar to the reference texts.

- **Precision vs. Recall:** The model has higher precision than recall across the board, which means it tends to generate relevant content well but might not always capture the full breadth of the reference texts.

- **Validation Loss:** At 1.026, the model's error rate is moderate, suggesting room for improvement but also indicating a decent understanding of the text generation task.


### Analysis of Differences - experiment 2 vs experiment 1

The numerical data presents a compelling case for the positive impact of the architectural optimizations introduced in Experiment 2. Future iterations of the model will likely continue to build on these improvements, aiming for even higher levels of performance in narrative text generation tasks.

- **BLEU Score Improvement:** The BLEU score saw an increase of approximately 92%, moving from 0.533 to 1.018. This significant jump indicates a near-doubling of accuracy in text generation compared to the reference texts.

- **Enhancements in ROUGE Metrics:** All ROUGE metrics observed substantial improvements. The ROUGE-1 F-measure increased by more than 202%, ROUGE-2 by approximately 186%, and ROUGE-L by nearly 198%. These metrics suggest a marked enhancement in the model's ability to replicate both the micro (words and two-word phrases) and macro (longer sequences) elements of the reference texts accurately.

- **Precision and Recall Analysis:** There is a considerable increase in ROUGE metrics for Experiment 2 implies improvements in both the relevance and completeness of the generated text relative to the references.

- **Reduction in Validation Loss:** The validation loss saw a slight decrease from 1.042 to 1.026. Although this change is modest, it still represents a positive shift towards reducing the model's average prediction error.



## Experiment 3: Input Sequence Modification
(Tues 20/02/2014 - branch prototype-end_to_end )

In this experiement we have changed `input_sequence` from the paper format ( premise, initial, orginal_ending, </s>, premise, couterfactual) to the format (premise, initial, original_ending, counterfactual)

### Results Table and Interpretation

| Metric                   | Experiment 2 Score    | Experiment 3 Score    | Difference          |
|--------------------------|-----------------------|-----------------------|---------------------|
| Average BLEU Score       | 1.018                 | 1.0175716876983643    | -0.0004283123016357 |
| ROUGE-1 F-measure        | 2.279                 | 2.2779605388641357    | -0.0010394611358643 |
| ROUGE-1 Precision        | 3.172                 | 3.1704189777374268    | -0.0015810222625732 |
| ROUGE-1 Recall           | 1.812                 | 1.81069815158844      | -0.00130184841156   |
| ROUGE-2 F-measure        | 1.813                 | 1.8134219646453857    |  0.0004219646453857 |
| ROUGE-2 Precision        | 2.580                 | 2.5816280841827393    |  0.0016280841827393 |
| ROUGE-2 Recall           | 1.425                 | 1.424911379814148     | -0.000088620185852  |
| ROUGE-L F-measure        | 2.201                 | 2.2031939029693604    |  0.0021939029693604 |
| ROUGE-L Precision        | 3.063                 | 3.0661745071411133    |  0.0031745071411133 |
| ROUGE-L Recall           | 1.750                 | 1.7513031959533691    |  0.0013031959533691 |
| ROUGE-Lsum F-measure     | 2.201                 | 2.2031939029693604    |  0.0021939029693604 |
| ROUGE-Lsum Precision     | 3.063                 | 3.0661745071411133    |  0.0031745071411133 |
| ROUGE-Lsum Recall        | 1.750                 | 1.7513031959533691    |  0.0013031959533691 |
| Validation Loss          | 1.026                 | 1.0790570974349976    |  0.0530570974349976 |

### Analysis of Differences - experiement 3 vs experiment 2

- **Average BLEU Score:** The difference is negligible, indicating that the change in input sequence format had an almost imperceptible impact on the model's accuracy.

- **ROUGE Scores:** The differences in ROUGE scores are minimal across the board. The slight variations observed (some positive, some negative) suggest minor shifts in the model's ability to capture both the individual words and longer coherent sequences of the reference texts. However, these shifts are so small that they might not be significant in practice.

- **Validation Loss:** The increase in validation loss is the most notable difference, suggesting a slight reduction in model performance with the modified input sequence format. While the absolute difference is small, it's the most significant change observed between the experiments, potentially indicating the separator token's role in model efficiency or understanding.

The table and analysis illustrate that the changes made in Experiment 3's input sequence format had minimal impact on most performance metrics, with the exception of a somewhat more pronounced increase in validation loss. This provides some insight into the effects of input formatting used in the original paper on the model performance.




---
inference: false
language:
- en
tags:
- text-generation
- pytorch
- causal-lm
license: apache-2.0
---

# GPT-4chan

## Model Description

GPT-4chan is a language model fine-tuned from [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) on 3.5 years worth of data from 4chan's _politically incorrect_ (/pol/) board. 

## Training data

GPT-4chan was fine-tuned on the dataset [Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board](https://zenodo.org/record/3606810).

## Training procedure

The model was trained for 1 epoch following [GPT-J's fine-tuning guide](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md).

## Intended Use

GPT-4chan is trained on anonymously posted and sparsely moderated discussions of political topics. Its intended use is to reproduce text according to the distribution of its input data. It may also be a useful tool to investigate discourse in such anonymous online communities. Lastly, it has potential applications in tasks suche as toxicity detection, as initial experiments show promising zero-shot results when comparing a string's likelihood under GPT-4chan to its likelihood under GPT-J 6B.

### How to use

The following is copied from the [Hugging Face documentation on GPT-J](https://huggingface.co/docs/transformers/main/en/model_doc/gptj#generation). Refer to the original for more details.

For inference parameters, we recommend a temperature of 0.8, along with either a top_p of 0.8 or a typical_p of 0.3.

For the float32 model (CPU):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("ykilcher/gpt-4chan")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

For the float16 model (GPU):
```python
from transformers import GPTJForCausalLM, AutoTokenizer
import torch

from transformers import GPTJForCausalLM
import torch

model = GPTJForCausalLM.from_pretrained(
    "ykilcher/gpt-4chan", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.cuda()

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

### Limitations and Biases

This is a statistical model. As such, it continues text as is likely under the distribution the model has learned from the training data. Outputs should not be interpreted as "correct", "truthful", or otherwise as anything more than a statistical function of the input. That being said, GPT-4chan does significantly outperform GPT-J (and GPT-3) on the [TruthfulQA Benchmark](https://arxiv.org/abs/2109.07958) that measures whether a language model is truthful in generating answers to questions.

The dataset is time- and domain-limited. It was collected from 2016 to 2019 on 4chan's _politically incorrect_ board. As such, political topics from that area will be overrepresented in the model's distribution, compared to other models (e.g. GPT-J 6B). Also, due to the very lax rules and anonymity of posters, a large part of the dataset contains offensive material. Thus, it is **very likely that the model will produce offensive outputs**, including but not limited to: toxicity, hate speech, racism, sexism, homo- and transphobia, xenophobia, and anti-semitism.

Due to the above limitations, it is strongly recommend to not deploy this model into a real-world environment unless its behavior is well-understood and explicit and strict limitations on the scope, impact, and duration of the deployment are enforced.

## Evaluation results


### Language Model Evaluation Harness

The following table compares GPT-J 6B to GPT-4chan on a subset of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
Differences exceeding standard errors are marked in the "Significant" column with a minus sign (-) indicating an advantage for GPT-J 6B and a plus sign (+) indicating an advantage for GPT-4chan.

<figure>

| Task                                                      | Metric          |    GPT-4chan |       stderr |     GPT-J-6B |       stderr | Significant   |
|:----------------------------------------------------------|:----------------|-------------:|-------------:|-------------:|-------------:|:--------------|
| copa                                                      | acc             |   0.85       |  0.035887    |   0.83       |  0.0377525   |               |
| blimp_only_npi_scope                                      | acc             |   0.712      |  0.0143269   |   0.787      |  0.0129537   | -             |
| hendrycksTest-conceptual_physics                          | acc             |   0.251064   |  0.028347    |   0.255319   |  0.0285049   |               |
| hendrycksTest-conceptual_physics                          | acc_norm        |   0.187234   |  0.0255016   |   0.191489   |  0.0257221   |               |
| hendrycksTest-high_school_mathematics                     | acc             |   0.248148   |  0.0263357   |   0.218519   |  0.0251958   | +             |
| hendrycksTest-high_school_mathematics                     | acc_norm        |   0.3        |  0.0279405   |   0.251852   |  0.0264661   | +             |
| blimp_sentential_negation_npi_scope                       | acc             |   0.734      |  0.01398     |   0.733      |  0.0139967   |               |
| hendrycksTest-high_school_european_history                | acc             |   0.278788   |  0.0350144   |   0.260606   |  0.0342774   |               |
| hendrycksTest-high_school_european_history                | acc_norm        |   0.315152   |  0.0362773   |   0.278788   |  0.0350144   | +             |
| blimp_wh_questions_object_gap                             | acc             |   0.841      |  0.0115695   |   0.835      |  0.0117436   |               |
| hendrycksTest-international_law                           | acc             |   0.214876   |  0.0374949   |   0.264463   |  0.0402619   | -             |
| hendrycksTest-international_law                           | acc_norm        |   0.438017   |  0.0452915   |   0.404959   |  0.0448114   |               |
| hendrycksTest-high_school_us_history                      | acc             |   0.323529   |  0.0328347   |   0.289216   |  0.0318223   | +             |
| hendrycksTest-high_school_us_history                      | acc_norm        |   0.323529   |  0.0328347   |   0.29902    |  0.0321333   |               |
| openbookqa                                                | acc             |   0.276      |  0.0200112   |   0.29       |  0.0203132   |               |
| openbookqa                                                | acc_norm        |   0.362      |  0.0215137   |   0.382      |  0.0217508   |               |
| blimp_causative                                           | acc             |   0.737      |  0.0139293   |   0.761      |  0.013493    | -             |
| record                                                    | f1              |   0.878443   |  0.00322394  |   0.885049   |  0.00314367  | -             |
| record                                                    | em              |   0.8702     |  0.003361    |   0.8765     |  0.00329027  | -             |
| blimp_determiner_noun_agreement_1                         | acc             |   0.996      |  0.00199699  |   0.995      |  0.00223159  |               |
| hendrycksTest-miscellaneous                               | acc             |   0.305236   |  0.0164677   |   0.274585   |  0.0159598   | +             |
| hendrycksTest-miscellaneous                               | acc_norm        |   0.269476   |  0.0158662   |   0.260536   |  0.015696    |               |
| hendrycksTest-virology                                    | acc             |   0.343373   |  0.0369658   |   0.349398   |  0.0371173   |               |
| hendrycksTest-virology                                    | acc_norm        |   0.331325   |  0.0366431   |   0.325301   |  0.0364717   |               |
| mathqa                                                    | acc             |   0.269012   |  0.00811786  |   0.267002   |  0.00809858  |               |
| mathqa                                                    | acc_norm        |   0.261642   |  0.00804614  |   0.270687   |  0.00813376  | -             |
| squad2                                                    | exact           |  10.6123     |  0           |  10.6207     |  0           | -             |
| squad2                                                    | f1              |  17.8734     |  0           |  17.7413     |  0           | +             |
| squad2                                                    | HasAns_exact    |  17.2571     |  0           |  15.5027     |  0           | +             |
| squad2                                                    | HasAns_f1       |  31.8        |  0           |  29.7643     |  0           | +             |
| squad2                                                    | NoAns_exact     |   3.98654    |  0           |   5.75273    |  0           | -             |
| squad2                                                    | NoAns_f1        |   3.98654    |  0           |   5.75273    |  0           | -             |
| squad2                                                    | best_exact      |  50.0716     |  0           |  50.0716     |  0           |               |
| squad2                                                    | best_f1         |  50.077      |  0           |  50.0778     |  0           | -             |
| mnli_mismatched                                           | acc             |   0.320586   |  0.00470696  |   0.376627   |  0.00488687  | -             |
| blimp_animate_subject_passive                             | acc             |   0.79       |  0.0128867   |   0.781      |  0.0130847   |               |
| blimp_determiner_noun_agreement_with_adj_irregular_1      | acc             |   0.834      |  0.0117721   |   0.878      |  0.0103549   | -             |
| qnli                                                      | acc             |   0.491305   |  0.00676439  |   0.513454   |  0.00676296  | -             |
| blimp_intransitive                                        | acc             |   0.806      |  0.0125108   |   0.858      |  0.0110435   | -             |
| ethics_cm                                                 | acc             |   0.512227   |  0.00802048  |   0.559846   |  0.00796521  | -             |
| hendrycksTest-high_school_computer_science                | acc             |   0.2        |  0.0402015   |   0.25       |  0.0435194   | -             |
| hendrycksTest-high_school_computer_science                | acc_norm        |   0.26       |  0.0440844   |   0.27       |  0.0446196   |               |
| iwslt17-ar-en                                             | bleu            |  21.4685     |  0.64825     |  20.7322     |  0.795602    | +             |
| iwslt17-ar-en                                             | chrf            |   0.452175   |  0.00498012  |   0.450919   |  0.00526515  |               |
| iwslt17-ar-en                                             | ter             |   0.733514   |  0.0201688   |   0.787631   |  0.0285488   | +             |
| hendrycksTest-security_studies                            | acc             |   0.391837   |  0.0312513   |   0.363265   |  0.0307891   |               |
| hendrycksTest-security_studies                            | acc_norm        |   0.285714   |  0.0289206   |   0.285714   |  0.0289206   |               |
| hendrycksTest-global_facts                                | acc             |   0.29       |  0.0456048   |   0.25       |  0.0435194   |               |
| hendrycksTest-global_facts                                | acc_norm        |   0.26       |  0.0440844   |   0.22       |  0.0416333   |               |
| anli_r1                                                   | acc             |   0.297      |  0.0144568   |   0.322      |  0.0147829   | -             |
| blimp_left_branch_island_simple_question                  | acc             |   0.884      |  0.0101315   |   0.867      |  0.0107437   | +             |
| hendrycksTest-astronomy                                   | acc             |   0.25       |  0.0352381   |   0.25       |  0.0352381   |               |
| hendrycksTest-astronomy                                   | acc_norm        |   0.348684   |  0.0387814   |   0.335526   |  0.038425    |               |
| mrpc                                                      | acc             |   0.536765   |  0.024717    |   0.683824   |  0.0230483   | -             |
| mrpc                                                      | f1              |   0.63301    |  0.0247985   |   0.812227   |  0.0162476   | -             |
| ethics_utilitarianism                                     | acc             |   0.525374   |  0.00720233  |   0.509775   |  0.00721024  | +             |
| blimp_determiner_noun_agreement_2                         | acc             |   0.99       |  0.003148    |   0.977      |  0.00474273  | +             |
| lambada_cloze                                             | ppl             | 388.123      | 13.1523      | 405.646      | 14.5519      | +             |
| lambada_cloze                                             | acc             |   0.0116437  |  0.00149456  |   0.0199884  |  0.00194992  | -             |
| truthfulqa_mc                                             | mc1             |   0.225214   |  0.0146232   |   0.201958   |  0.014054    | +             |
| truthfulqa_mc                                             | mc2             |   0.371625   |  0.0136558   |   0.359537   |  0.0134598   |               |
| blimp_wh_vs_that_with_gap_long_distance                   | acc             |   0.441      |  0.0157088   |   0.342      |  0.0150087   | +             |
| hendrycksTest-business_ethics                             | acc             |   0.28       |  0.0451261   |   0.29       |  0.0456048   |               |
| hendrycksTest-business_ethics                             | acc_norm        |   0.29       |  0.0456048   |   0.3        |  0.0460566   |               |
| arithmetic_3ds                                            | acc             |   0.0065     |  0.00179736  |   0.046      |  0.0046854   | -             |
| blimp_determiner_noun_agreement_with_adjective_1          | acc             |   0.988      |  0.00344498  |   0.978      |  0.00464086  | +             |
| hendrycksTest-moral_disputes                              | acc             |   0.277457   |  0.0241057   |   0.283237   |  0.0242579   |               |
| hendrycksTest-moral_disputes                              | acc_norm        |   0.309249   |  0.0248831   |   0.32659    |  0.0252483   |               |
| arithmetic_2da                                            | acc             |   0.0455     |  0.00466109  |   0.2405     |  0.00955906  | -             |
| qa4mre_2011                                               | acc             |   0.425      |  0.0453163   |   0.458333   |  0.0456755   |               |
| qa4mre_2011                                               | acc_norm        |   0.558333   |  0.0455219   |   0.533333   |  0.045733    |               |
| blimp_regular_plural_subject_verb_agreement_1             | acc             |   0.966      |  0.00573384  |   0.968      |  0.00556839  |               |
| hendrycksTest-human_sexuality                             | acc             |   0.389313   |  0.0427649   |   0.396947   |  0.0429114   |               |
| hendrycksTest-human_sexuality                             | acc_norm        |   0.305344   |  0.0403931   |   0.343511   |  0.0416498   |               |
| blimp_passive_1                                           | acc             |   0.878      |  0.0103549   |   0.885      |  0.0100934   |               |
| blimp_drop_argument                                       | acc             |   0.784      |  0.0130197   |   0.823      |  0.0120755   | -             |
| hendrycksTest-high_school_microeconomics                  | acc             |   0.260504   |  0.0285103   |   0.277311   |  0.0290794   |               |
| hendrycksTest-high_school_microeconomics                  | acc_norm        |   0.390756   |  0.0316938   |   0.39916    |  0.0318111   |               |
| hendrycksTest-us_foreign_policy                           | acc             |   0.32       |  0.0468826   |   0.34       |  0.0476095   |               |
| hendrycksTest-us_foreign_policy                           | acc_norm        |   0.4        |  0.0492366   |   0.35       |  0.0479372   | +             |
| blimp_ellipsis_n_bar_1                                    | acc             |   0.846      |  0.0114199   |   0.841      |  0.0115695   |               |
| hendrycksTest-high_school_physics                         | acc             |   0.264901   |  0.0360304   |   0.271523   |  0.0363133   |               |
| hendrycksTest-high_school_physics                         | acc_norm        |   0.284768   |  0.0368488   |   0.271523   |  0.0363133   |               |
| qa4mre_2013                                               | acc             |   0.362676   |  0.028579    |   0.401408   |  0.0291384   | -             |
| qa4mre_2013                                               | acc_norm        |   0.387324   |  0.0289574   |   0.383803   |  0.0289082   |               |
| blimp_wh_vs_that_no_gap                                   | acc             |   0.963      |  0.00597216  |   0.969      |  0.00548353  | -             |
| headqa_es                                                 | acc             |   0.238877   |  0.00814442  |   0.251276   |  0.0082848   | -             |
| headqa_es                                                 | acc_norm        |   0.290664   |  0.00867295  |   0.286652   |  0.00863721  |               |
| blimp_sentential_subject_island                           | acc             |   0.359      |  0.0151773   |   0.421      |  0.0156206   | -             |
| hendrycksTest-philosophy                                  | acc             |   0.241158   |  0.0242966   |   0.26045    |  0.0249267   |               |
| hendrycksTest-philosophy                                  | acc_norm        |   0.327974   |  0.0266644   |   0.334405   |  0.0267954   |               |
| hendrycksTest-elementary_mathematics                      | acc             |   0.248677   |  0.0222618   |   0.251323   |  0.0223405   |               |
| hendrycksTest-elementary_mathematics                      | acc_norm        |   0.275132   |  0.0230001   |   0.26455    |  0.0227175   |               |
| math_geometry                                             | acc             |   0.0187891  |  0.00621042  |   0.0104384  |  0.00464863  | +             |
| blimp_wh_questions_subject_gap_long_distance              | acc             |   0.886      |  0.0100551   |   0.883      |  0.0101693   |               |
| hendrycksTest-college_physics                             | acc             |   0.205882   |  0.0402338   |   0.205882   |  0.0402338   |               |
| hendrycksTest-college_physics                             | acc_norm        |   0.22549    |  0.0415831   |   0.245098   |  0.0428011   |               |
| hellaswag                                                 | acc             |   0.488747   |  0.00498852  |   0.49532    |  0.00498956  | -             |
| hellaswag                                                 | acc_norm        |   0.648277   |  0.00476532  |   0.66202    |  0.00472055  | -             |
| hendrycksTest-logical_fallacies                           | acc             |   0.269939   |  0.0348783   |   0.294479   |  0.0358117   |               |
| hendrycksTest-logical_fallacies                           | acc_norm        |   0.343558   |  0.0373113   |   0.355828   |  0.0376152   |               |
| hendrycksTest-machine_learning                            | acc             |   0.339286   |  0.0449395   |   0.223214   |  0.039523    | +             |
| hendrycksTest-machine_learning                            | acc_norm        |   0.205357   |  0.0383424   |   0.178571   |  0.0363521   |               |
| hendrycksTest-high_school_psychology                      | acc             |   0.286239   |  0.0193794   |   0.273394   |  0.0191093   |               |
| hendrycksTest-high_school_psychology                      | acc_norm        |   0.266055   |  0.018946    |   0.269725   |  0.0190285   |               |
| prost                                                     | acc             |   0.256298   |  0.00318967  |   0.268254   |  0.00323688  | -             |
| prost                                                     | acc_norm        |   0.280156   |  0.00328089  |   0.274658   |  0.00326093  | +             |
| blimp_determiner_noun_agreement_with_adj_irregular_2      | acc             |   0.898      |  0.00957537  |   0.916      |  0.00877616  | -             |
| wnli                                                      | acc             |   0.43662    |  0.0592794   |   0.464789   |  0.0596131   |               |
| hendrycksTest-professional_law                            | acc             |   0.284876   |  0.0115278   |   0.273794   |  0.0113886   |               |
| hendrycksTest-professional_law                            | acc_norm        |   0.301825   |  0.0117244   |   0.292699   |  0.0116209   |               |
| math_algebra                                              | acc             |   0.0126369  |  0.00324352  |   0.0117944  |  0.00313487  |               |
| wikitext                                                  | word_perplexity |  11.4687     |  0           |  10.8819     |  0           | -             |
| wikitext                                                  | byte_perplexity |   1.5781     |  0           |   1.56268    |  0           | -             |
| wikitext                                                  | bits_per_byte   |   0.658188   |  0           |   0.644019   |  0           | -             |
| anagrams1                                                 | acc             |   0.0125     |  0.00111108  |   0.0008     |  0.000282744 | +             |
| math_prealgebra                                           | acc             |   0.0195178  |  0.00469003  |   0.0126292  |  0.00378589  | +             |
| blimp_principle_A_domain_2                                | acc             |   0.887      |  0.0100166   |   0.889      |  0.0099387   |               |
| cycle_letters                                             | acc             |   0.0331     |  0.00178907  |   0.0026     |  0.000509264 | +             |
| hendrycksTest-college_mathematics                         | acc             |   0.26       |  0.0440844   |   0.26       |  0.0440844   |               |
| hendrycksTest-college_mathematics                         | acc_norm        |   0.31       |  0.0464823   |   0.4        |  0.0492366   | -             |
| arithmetic_1dc                                            | acc             |   0.077      |  0.00596266  |   0.089      |  0.00636866  | -             |
| arithmetic_4da                                            | acc             |   0.0005     |  0.0005      |   0.007      |  0.00186474  | -             |
| triviaqa                                                  | acc             |   0.150888   |  0.00336543  |   0.167418   |  0.00351031  | -             |
| boolq                                                     | acc             |   0.673394   |  0.00820236  |   0.655352   |  0.00831224  | +             |
| random_insertion                                          | acc             |   0.0004     |  0.00019997  |   0          |  0           | +             |
| qa4mre_2012                                               | acc             |   0.4        |  0.0388514   |   0.4125     |  0.0390407   |               |
| qa4mre_2012                                               | acc_norm        |   0.4625     |  0.0395409   |   0.50625    |  0.0396495   | -             |
| math_asdiv                                                | acc             |   0.00997831 |  0.00207066  |   0.00563991 |  0.00156015  | +             |
| hendrycksTest-moral_scenarios                             | acc             |   0.236872   |  0.0142196   |   0.236872   |  0.0142196   |               |
| hendrycksTest-moral_scenarios                             | acc_norm        |   0.272626   |  0.0148934   |   0.272626   |  0.0148934   |               |
| hendrycksTest-high_school_geography                       | acc             |   0.247475   |  0.0307463   |   0.20202    |  0.0286062   | +             |
| hendrycksTest-high_school_geography                       | acc_norm        |   0.287879   |  0.0322588   |   0.292929   |  0.032425    |               |
| gsm8k                                                     | acc             |   0          |  0           |   0          |  0           |               |
| blimp_existential_there_object_raising                    | acc             |   0.812      |  0.0123616   |   0.792      |  0.0128414   | +             |
| blimp_superlative_quantifiers_2                           | acc             |   0.917      |  0.00872853  |   0.865      |  0.0108117   | +             |
| hendrycksTest-college_chemistry                           | acc             |   0.28       |  0.0451261   |   0.24       |  0.0429235   |               |
| hendrycksTest-college_chemistry                           | acc_norm        |   0.31       |  0.0464823   |   0.28       |  0.0451261   |               |
| blimp_existential_there_quantifiers_2                     | acc             |   0.545      |  0.0157551   |   0.383      |  0.0153801   | +             |
| hendrycksTest-abstract_algebra                            | acc             |   0.17       |  0.0377525   |   0.26       |  0.0440844   | -             |
| hendrycksTest-abstract_algebra                            | acc_norm        |   0.26       |  0.0440844   |   0.3        |  0.0460566   |               |
| hendrycksTest-professional_psychology                     | acc             |   0.26634    |  0.0178832   |   0.28268    |  0.0182173   |               |
| hendrycksTest-professional_psychology                     | acc_norm        |   0.256536   |  0.0176678   |   0.259804   |  0.0177409   |               |
| ethics_virtue                                             | acc             |   0.249849   |  0.00613847  |   0.200201   |  0.00567376  | +             |
| ethics_virtue                                             | em              |   0.0040201  |  0           |   0          |  0           | +             |
| arithmetic_5da                                            | acc             |   0          |  0           |   0.0005     |  0.0005      | -             |
| mutual                                                    | r@1             |   0.455982   |  0.0167421   |   0.468397   |  0.0167737   |               |
| mutual                                                    | r@2             |   0.732506   |  0.0148796   |   0.735892   |  0.0148193   |               |
| mutual                                                    | mrr             |   0.675226   |  0.0103132   |   0.682186   |  0.0103375   |               |
| blimp_irregular_past_participle_verbs                     | acc             |   0.869      |  0.0106749   |   0.876      |  0.0104275   |               |
| ethics_deontology                                         | acc             |   0.497775   |  0.00833904  |   0.523637   |  0.0083298   | -             |
| ethics_deontology                                         | em              |   0.00333704 |  0           |   0.0355951  |  0           | -             |
| blimp_transitive                                          | acc             |   0.818      |  0.0122076   |   0.855      |  0.01114     | -             |
| hendrycksTest-college_computer_science                    | acc             |   0.29       |  0.0456048   |   0.27       |  0.0446196   |               |
| hendrycksTest-college_computer_science                    | acc_norm        |   0.27       |  0.0446196   |   0.26       |  0.0440844   |               |
| hendrycksTest-professional_medicine                       | acc             |   0.283088   |  0.0273659   |   0.272059   |  0.027033    |               |
| hendrycksTest-professional_medicine                       | acc_norm        |   0.279412   |  0.0272572   |   0.261029   |  0.0266793   |               |
| sciq                                                      | acc             |   0.895      |  0.00969892  |   0.915      |  0.00882343  | -             |
| sciq                                                      | acc_norm        |   0.869      |  0.0106749   |   0.874      |  0.0104992   |               |
| blimp_anaphor_number_agreement                            | acc             |   0.993      |  0.00263779  |   0.995      |  0.00223159  |               |
| blimp_wh_questions_subject_gap                            | acc             |   0.925      |  0.00833333  |   0.913      |  0.00891687  | +             |
| blimp_wh_vs_that_with_gap                                 | acc             |   0.482      |  0.015809    |   0.429      |  0.015659    | +             |
| math_num_theory                                           | acc             |   0.0351852  |  0.00793611  |   0.0203704  |  0.00608466  | +             |
| blimp_complex_NP_island                                   | acc             |   0.538      |  0.0157735   |   0.535      |  0.0157805   |               |
| blimp_expletive_it_object_raising                         | acc             |   0.777      |  0.0131698   |   0.78       |  0.0131062   |               |
| lambada_mt_en                                             | ppl             |   4.62504    |  0.10549     |   4.10224    |  0.0884971   | -             |
| lambada_mt_en                                             | acc             |   0.648554   |  0.00665142  |   0.682127   |  0.00648741  | -             |
| hendrycksTest-formal_logic                                | acc             |   0.309524   |  0.0413491   |   0.34127    |  0.042408    |               |
| hendrycksTest-formal_logic                                | acc_norm        |   0.325397   |  0.041906    |   0.325397   |  0.041906    |               |
| blimp_matrix_question_npi_licensor_present                | acc             |   0.663      |  0.0149551   |   0.727      |  0.014095    | -             |
| blimp_superlative_quantifiers_1                           | acc             |   0.791      |  0.0128641   |   0.871      |  0.0106053   | -             |
| lambada_mt_de                                             | ppl             |  89.7905     |  5.30301     |  82.2416     |  4.88447     | -             |
| lambada_mt_de                                             | acc             |   0.312245   |  0.0064562   |   0.312827   |  0.00645948  |               |
| hendrycksTest-computer_security                           | acc             |   0.37       |  0.0485237   |   0.27       |  0.0446196   | +             |
| hendrycksTest-computer_security                           | acc_norm        |   0.37       |  0.0485237   |   0.33       |  0.0472582   |               |
| ethics_justice                                            | acc             |   0.501479   |  0.00961712  |   0.526627   |  0.00960352  | -             |
| ethics_justice                                            | em              |   0          |  0           |   0.0251479  |  0           | -             |
| blimp_principle_A_reconstruction                          | acc             |   0.296      |  0.0144427   |   0.444      |  0.0157198   | -             |
| blimp_existential_there_subject_raising                   | acc             |   0.877      |  0.0103913   |   0.875      |  0.0104635   |               |
| math_precalc                                              | acc             |   0.014652   |  0.00514689  |   0.0018315  |  0.0018315   | +             |
| qasper                                                    | f1_yesno        |   0.632997   |  0.032868    |   0.666667   |  0.0311266   | -             |
| qasper                                                    | f1_abstractive  |   0.113489   |  0.00729073  |   0.118383   |  0.00692993  |               |
| cb                                                        | acc             |   0.196429   |  0.0535714   |   0.357143   |  0.0646096   | -             |
| cb                                                        | f1              |   0.149038   |  0           |   0.288109   |  0           | -             |
| blimp_animate_subject_trans                               | acc             |   0.858      |  0.0110435   |   0.868      |  0.0107094   |               |
| hendrycksTest-high_school_statistics                      | acc             |   0.310185   |  0.031547    |   0.291667   |  0.0309987   |               |
| hendrycksTest-high_school_statistics                      | acc_norm        |   0.361111   |  0.0327577   |   0.314815   |  0.0316747   | +             |
| blimp_irregular_plural_subject_verb_agreement_2           | acc             |   0.881      |  0.0102442   |   0.919      |  0.00863212  | -             |
| lambada_mt_es                                             | ppl             |  92.1172     |  5.05064     |  83.6696     |  4.57489     | -             |
| lambada_mt_es                                             | acc             |   0.322337   |  0.00651139  |   0.326994   |  0.00653569  |               |
| anli_r2                                                   | acc             |   0.327      |  0.0148422   |   0.337      |  0.0149551   |               |
| hendrycksTest-nutrition                                   | acc             |   0.346405   |  0.0272456   |   0.346405   |  0.0272456   |               |
| hendrycksTest-nutrition                                   | acc_norm        |   0.385621   |  0.0278707   |   0.401961   |  0.0280742   |               |
| anli_r3                                                   | acc             |   0.336667   |  0.0136476   |   0.3525     |  0.0137972   | -             |
| blimp_regular_plural_subject_verb_agreement_2             | acc             |   0.897      |  0.00961683  |   0.916      |  0.00877616  | -             |
| blimp_tough_vs_raising_2                                  | acc             |   0.826      |  0.0119945   |   0.857      |  0.0110758   | -             |
| mnli                                                      | acc             |   0.316047   |  0.00469317  |   0.374733   |  0.00488619  | -             |
| drop                                                      | em              |   0.0595638  |  0.00242379  |   0.0228607  |  0.0015306   | +             |
| drop                                                      | f1              |   0.120355   |  0.00270951  |   0.103871   |  0.00219977  | +             |
| blimp_determiner_noun_agreement_with_adj_2                | acc             |   0.95       |  0.00689547  |   0.936      |  0.00774364  | +             |
| arithmetic_2dm                                            | acc             |   0.061      |  0.00535293  |   0.14       |  0.00776081  | -             |
| blimp_determiner_noun_agreement_irregular_2               | acc             |   0.93       |  0.00807249  |   0.932      |  0.00796489  |               |
| lambada                                                   | ppl             |   4.62504    |  0.10549     |   4.10224    |  0.0884971   | -             |
| lambada                                                   | acc             |   0.648554   |  0.00665142  |   0.682127   |  0.00648741  | -             |
| arithmetic_3da                                            | acc             |   0.007      |  0.00186474  |   0.0865     |  0.00628718  | -             |
| blimp_irregular_past_participle_adjectives                | acc             |   0.947      |  0.00708811  |   0.956      |  0.00648892  | -             |
| hendrycksTest-college_biology                             | acc             |   0.201389   |  0.0335365   |   0.284722   |  0.0377381   | -             |
| hendrycksTest-college_biology                             | acc_norm        |   0.222222   |  0.0347659   |   0.270833   |  0.0371618   | -             |
| headqa_en                                                 | acc             |   0.324945   |  0.00894582  |   0.335522   |  0.00901875  | -             |
| headqa_en                                                 | acc_norm        |   0.375638   |  0.00925014  |   0.383297   |  0.00928648  |               |
| blimp_determiner_noun_agreement_irregular_1               | acc             |   0.912      |  0.00896305  |   0.944      |  0.0072744   | -             |
| blimp_existential_there_quantifiers_1                     | acc             |   0.985      |  0.00384575  |   0.981      |  0.00431945  |               |
| blimp_inchoative                                          | acc             |   0.653      |  0.0150605   |   0.683      |  0.0147217   | -             |
| mutual_plus                                               | r@1             |   0.395034   |  0.0164328   |   0.409707   |  0.016531    |               |
| mutual_plus                                               | r@2             |   0.674944   |  0.015745    |   0.680587   |  0.0156728   |               |
| mutual_plus                                               | mrr             |   0.632713   |  0.0103391   |   0.640801   |  0.0104141   |               |
| blimp_tough_vs_raising_1                                  | acc             |   0.736      |  0.0139463   |   0.734      |  0.01398     |               |
| winogrande                                                | acc             |   0.636148   |  0.0135215   |   0.640884   |  0.0134831   |               |
| race                                                      | acc             |   0.374163   |  0.0149765   |   0.37512    |  0.0149842   |               |
| blimp_irregular_plural_subject_verb_agreement_1           | acc             |   0.908      |  0.00914438  |   0.918      |  0.00868052  | -             |
| hendrycksTest-high_school_macroeconomics                  | acc             |   0.284615   |  0.0228783   |   0.284615   |  0.0228783   |               |
| hendrycksTest-high_school_macroeconomics                  | acc_norm        |   0.284615   |  0.0228783   |   0.276923   |  0.022688    |               |
| blimp_adjunct_island                                      | acc             |   0.888      |  0.00997775  |   0.902      |  0.00940662  | -             |
| hendrycksTest-high_school_chemistry                       | acc             |   0.236453   |  0.0298961   |   0.211823   |  0.028749    |               |
| hendrycksTest-high_school_chemistry                       | acc_norm        |   0.300493   |  0.032258    |   0.29064    |  0.0319474   |               |
| arithmetic_2ds                                            | acc             |   0.051      |  0.00492053  |   0.218      |  0.00923475  | -             |
| blimp_principle_A_case_2                                  | acc             |   0.955      |  0.00655881  |   0.953      |  0.00669596  |               |
| blimp_only_npi_licensor_present                           | acc             |   0.926      |  0.00828206  |   0.953      |  0.00669596  | -             |
| math_counting_and_prob                                    | acc             |   0.0274262  |  0.00750954  |   0.0021097  |  0.0021097   | +             |
| cola                                                      | mcc             |  -0.0854256  |  0.0304519   |  -0.0504508  |  0.0251594   | -             |
| webqs                                                     | acc             |   0.023622   |  0.00336987  |   0.0226378  |  0.00330058  |               |
| arithmetic_4ds                                            | acc             |   0.0005     |  0.0005      |   0.0055     |  0.00165416  | -             |
| blimp_wh_vs_that_no_gap_long_distance                     | acc             |   0.94       |  0.00751375  |   0.939      |  0.00757208  |               |
| pile_bookcorpus2                                          | word_perplexity |  28.7786     |  0           |  27.0559     |  0           | -             |
| pile_bookcorpus2                                          | byte_perplexity |   1.79969    |  0           |   1.78037    |  0           | -             |
| pile_bookcorpus2                                          | bits_per_byte   |   0.847751   |  0           |   0.832176   |  0           | -             |
| blimp_sentential_negation_npi_licensor_present            | acc             |   0.994      |  0.00244335  |   0.982      |  0.00420639  | +             |
| hendrycksTest-high_school_government_and_politics         | acc             |   0.274611   |  0.0322102   |   0.227979   |  0.0302769   | +             |
| hendrycksTest-high_school_government_and_politics         | acc_norm        |   0.259067   |  0.0316188   |   0.248705   |  0.0311958   |               |
| blimp_ellipsis_n_bar_2                                    | acc             |   0.937      |  0.00768701  |   0.916      |  0.00877616  | +             |
| hendrycksTest-clinical_knowledge                          | acc             |   0.283019   |  0.0277242   |   0.267925   |  0.0272573   |               |
| hendrycksTest-clinical_knowledge                          | acc_norm        |   0.343396   |  0.0292245   |   0.316981   |  0.0286372   |               |
| mc_taco                                                   | em              |   0.125375   |  0           |   0.132883   |  0           | -             |
| mc_taco                                                   | f1              |   0.487131   |  0           |   0.499712   |  0           | -             |
| wsc                                                       | acc             |   0.365385   |  0.0474473   |   0.365385   |  0.0474473   |               |
| hendrycksTest-college_medicine                            | acc             |   0.231214   |  0.0321474   |   0.190751   |  0.0299579   | +             |
| hendrycksTest-college_medicine                            | acc_norm        |   0.289017   |  0.0345643   |   0.265896   |  0.0336876   |               |
| hendrycksTest-high_school_world_history                   | acc             |   0.295359   |  0.0296963   |   0.2827     |  0.0293128   |               |
| hendrycksTest-high_school_world_history                   | acc_norm        |   0.312236   |  0.0301651   |   0.312236   |  0.0301651   |               |
| hendrycksTest-anatomy                                     | acc             |   0.296296   |  0.0394462   |   0.281481   |  0.03885     |               |
| hendrycksTest-anatomy                                     | acc_norm        |   0.288889   |  0.0391545   |   0.266667   |  0.0382017   |               |
| hendrycksTest-jurisprudence                               | acc             |   0.25       |  0.0418609   |   0.277778   |  0.0433004   |               |
| hendrycksTest-jurisprudence                               | acc_norm        |   0.416667   |  0.0476608   |   0.425926   |  0.0478034   |               |
| logiqa                                                    | acc             |   0.193548   |  0.0154963   |   0.211982   |  0.016031    | -             |
| logiqa                                                    | acc_norm        |   0.281106   |  0.0176324   |   0.291859   |  0.0178316   |               |
| ethics_utilitarianism_original                            | acc             |   0.767679   |  0.00609112  |   0.941556   |  0.00338343  | -             |
| blimp_principle_A_c_command                               | acc             |   0.827      |  0.0119672   |   0.81       |  0.0124119   | +             |
| blimp_coordinate_structure_constraint_complex_left_branch | acc             |   0.794      |  0.0127956   |   0.764      |  0.0134345   | +             |
| arithmetic_5ds                                            | acc             |   0          |  0           |   0          |  0           |               |
| lambada_mt_it                                             | ppl             |  96.8846     |  5.80902     |  86.66       |  5.1869      | -             |
| lambada_mt_it                                             | acc             |   0.328158   |  0.00654165  |   0.336891   |  0.0065849   | -             |
| wsc273                                                    | acc             |   0.827839   |  0.0228905   |   0.827839   |  0.0228905   |               |
| blimp_coordinate_structure_constraint_object_extraction   | acc             |   0.852      |  0.0112349   |   0.876      |  0.0104275   | -             |
| blimp_principle_A_domain_3                                | acc             |   0.79       |  0.0128867   |   0.819      |  0.0121814   | -             |
| blimp_left_branch_island_echo_question                    | acc             |   0.638      |  0.0152048   |   0.519      |  0.0158079   | +             |
| rte                                                       | acc             |   0.534296   |  0.0300256   |   0.548736   |  0.0299531   |               |
| blimp_passive_2                                           | acc             |   0.892      |  0.00982     |   0.899      |  0.00953362  |               |
| hendrycksTest-electrical_engineering                      | acc             |   0.344828   |  0.0396093   |   0.358621   |  0.0399663   |               |
| hendrycksTest-electrical_engineering                      | acc_norm        |   0.372414   |  0.0402873   |   0.372414   |  0.0402873   |               |
| sst                                                       | acc             |   0.626147   |  0.0163938   |   0.493119   |  0.0169402   | +             |
| blimp_npi_present_1                                       | acc             |   0.565      |  0.0156851   |   0.576      |  0.0156355   |               |
| piqa                                                      | acc             |   0.739391   |  0.0102418   |   0.754081   |  0.0100473   | -             |
| piqa                                                      | acc_norm        |   0.755169   |  0.0100323   |   0.761697   |  0.00994033  |               |
| hendrycksTest-professional_accounting                     | acc             |   0.312057   |  0.0276401   |   0.265957   |  0.0263581   | +             |
| hendrycksTest-professional_accounting                     | acc_norm        |   0.27305    |  0.0265779   |   0.22695    |  0.0249871   | +             |
| arc_challenge                                             | acc             |   0.325085   |  0.0136881   |   0.337884   |  0.013822    |               |
| arc_challenge                                             | acc_norm        |   0.352389   |  0.0139601   |   0.366041   |  0.0140772   |               |
| hendrycksTest-econometrics                                | acc             |   0.263158   |  0.0414244   |   0.245614   |  0.0404934   |               |
| hendrycksTest-econometrics                                | acc_norm        |   0.254386   |  0.0409699   |   0.27193    |  0.0418577   |               |
| headqa                                                    | acc             |   0.238877   |  0.00814442  |   0.251276   |  0.0082848   | -             |
| headqa                                                    | acc_norm        |   0.290664   |  0.00867295  |   0.286652   |  0.00863721  |               |
| wic                                                       | acc             |   0.482759   |  0.0197989   |   0.5        |  0.0198107   |               |
| hendrycksTest-high_school_biology                         | acc             |   0.270968   |  0.0252844   |   0.251613   |  0.024686    |               |
| hendrycksTest-high_school_biology                         | acc_norm        |   0.274194   |  0.0253781   |   0.283871   |  0.0256494   |               |
| hendrycksTest-management                                  | acc             |   0.281553   |  0.0445325   |   0.23301    |  0.0418583   | +             |
| hendrycksTest-management                                  | acc_norm        |   0.291262   |  0.0449868   |   0.320388   |  0.0462028   |               |
| blimp_npi_present_2                                       | acc             |   0.645      |  0.0151395   |   0.664      |  0.0149441   | -             |
| hendrycksTest-prehistory                                  | acc             |   0.265432   |  0.0245692   |   0.243827   |  0.0238919   |               |
| hendrycksTest-prehistory                                  | acc_norm        |   0.225309   |  0.0232462   |   0.219136   |  0.0230167   |               |
| hendrycksTest-world_religions                             | acc             |   0.321637   |  0.0358253   |   0.333333   |  0.0361551   |               |
| hendrycksTest-world_religions                             | acc_norm        |   0.397661   |  0.0375364   |   0.380117   |  0.0372297   |               |
| math_intermediate_algebra                                 | acc             |   0.00996678 |  0.00330749  |   0.00332226 |  0.00191598  | +             |
| anagrams2                                                 | acc             |   0.0347     |  0.00183028  |   0.0055     |  0.000739615 | +             |
| arc_easy                                                  | acc             |   0.647306   |  0.00980442  |   0.669613   |  0.00965143  | -             |
| arc_easy                                                  | acc_norm        |   0.609848   |  0.0100091   |   0.622896   |  0.00994504  | -             |
| blimp_anaphor_gender_agreement                            | acc             |   0.993      |  0.00263779  |   0.994      |  0.00244335  |               |
| hendrycksTest-marketing                                   | acc             |   0.311966   |  0.0303515   |   0.307692   |  0.0302364   |               |
| hendrycksTest-marketing                                   | acc_norm        |   0.34188    |  0.031075    |   0.294872   |  0.0298726   | +             |
| blimp_principle_A_domain_1                                | acc             |   0.997      |  0.00173032  |   0.997      |  0.00173032  |               |
| blimp_wh_island                                           | acc             |   0.856      |  0.011108    |   0.852      |  0.0112349   |               |
| hendrycksTest-sociology                                   | acc             |   0.303483   |  0.0325101   |   0.278607   |  0.0317006   |               |
| hendrycksTest-sociology                                   | acc_norm        |   0.298507   |  0.0323574   |   0.318408   |  0.0329412   |               |
| blimp_distractor_agreement_relative_clause                | acc             |   0.774      |  0.0132325   |   0.719      |  0.0142212   | +             |
| truthfulqa_gen                                            | bleurt_max      |  -0.811655   |  0.0180743   |  -0.814228   |  0.0172128   |               |
| truthfulqa_gen                                            | bleurt_acc      |   0.395349   |  0.0171158   |   0.329253   |  0.0164513   | +             |
| truthfulqa_gen                                            | bleurt_diff     |  -0.0488385  |  0.0204525   |  -0.185905   |  0.0169617   | +             |
| truthfulqa_gen                                            | bleu_max        |  20.8747     |  0.717003    |  20.2238     |  0.711772    |               |
| truthfulqa_gen                                            | bleu_acc        |   0.330477   |  0.0164668   |   0.281518   |  0.015744    | +             |
| truthfulqa_gen                                            | bleu_diff       |  -2.12856    |  0.832693    |  -6.66121    |  0.719366    | +             |
| truthfulqa_gen                                            | rouge1_max      |  47.0293     |  0.962404    |  45.3457     |  0.89238     | +             |
| truthfulqa_gen                                            | rouge1_acc      |   0.341493   |  0.0166007   |   0.257038   |  0.0152981   | +             |
| truthfulqa_gen                                            | rouge1_diff     |  -2.29454    |  1.2086      | -10.1049     |  0.8922      | +             |
| truthfulqa_gen                                            | rouge2_max      |  31.0617     |  1.08725     |  28.7438     |  0.981282    | +             |
| truthfulqa_gen                                            | rouge2_acc      |   0.247246   |  0.0151024   |   0.201958   |  0.014054    | +             |
| truthfulqa_gen                                            | rouge2_diff     |  -2.84021    |  1.28749     | -11.0916     |  1.01664     | +             |
| truthfulqa_gen                                            | rougeL_max      |  44.6463     |  0.966119    |  42.6116     |  0.893252    | +             |
| truthfulqa_gen                                            | rougeL_acc      |   0.334149   |  0.0165125   |   0.24235    |  0.0150007   | +             |
| truthfulqa_gen                                            | rougeL_diff     |  -2.50853    |  1.22016     | -10.4299     |  0.904205    | +             |
| hendrycksTest-public_relations                            | acc             |   0.3        |  0.0438931   |   0.281818   |  0.0430912   |               |
| hendrycksTest-public_relations                            | acc_norm        |   0.190909   |  0.0376443   |   0.163636   |  0.0354343   |               |
| blimp_distractor_agreement_relational_noun                | acc             |   0.859      |  0.0110109   |   0.833      |  0.0118004   | +             |
| lambada_mt_fr                                             | ppl             |  57.0379     |  3.15719     |  51.7313     |  2.90272     | -             |
| lambada_mt_fr                                             | acc             |   0.388512   |  0.0067906   |   0.40947    |  0.00685084  | -             |
| blimp_principle_A_case_1                                  | acc             |   1          |  0           |   1          |  0           |               |
| hendrycksTest-medical_genetics                            | acc             |   0.37       |  0.0485237   |   0.31       |  0.0464823   | +             |
| hendrycksTest-medical_genetics                            | acc_norm        |   0.41       |  0.0494311   |   0.39       |  0.0490207   |               |
| qqp                                                       | acc             |   0.364383   |  0.00239348  |   0.383626   |  0.00241841  | -             |
| qqp                                                       | f1              |   0.516391   |  0.00263674  |   0.451222   |  0.00289696  | +             |
| iwslt17-en-ar                                             | bleu            |   2.35563    |  0.188638    |   4.98225    |  0.275369    | -             |
| iwslt17-en-ar                                             | chrf            |   0.140912   |  0.00503101  |   0.277708   |  0.00415432  | -             |
| iwslt17-en-ar                                             | ter             |   1.0909     |  0.0122111   |   0.954701   |  0.0126737   | -             |
| multirc                                                   | acc             |   0.0409234  |  0.00642087  |   0.0178384  |  0.00428994  | +             |
| hendrycksTest-human_aging                                 | acc             |   0.264574   |  0.0296051   |   0.264574   |  0.0296051   |               |
| hendrycksTest-human_aging                                 | acc_norm        |   0.197309   |  0.0267099   |   0.237668   |  0.0285681   | -             |
| reversed_words                                            | acc             |   0.0003     |  0.000173188 |   0          |  0           | +             |
<figcaption><p>Some results are missing due to errors or computational constraints.</p>
</figcaption></figure>


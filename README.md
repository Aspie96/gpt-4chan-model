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

| Task                                                      | Metric          |     GPT-J-6B |       stderr |    GPT-4chan |       stderr | Significant   |
|:----------------------------------------------------------|:----------------|-------------:|-------------:|-------------:|-------------:|:--------------|
| copa                                                      | acc             |   0.83       |  0.0377525   |   0.85       |  0.035887    |               |
| blimp_only_npi_scope                                      | acc             |   0.787      |  0.0129537   |   0.712      |  0.0143269   | -             |
| hendrycksTest-conceptual_physics                          | acc             |   0.255319   |  0.0285049   |   0.251064   |  0.028347    |               |
| hendrycksTest-conceptual_physics                          | acc_norm        |   0.191489   |  0.0257221   |   0.187234   |  0.0255016   |               |
| hendrycksTest-high_school_mathematics                     | acc             |   0.218519   |  0.0251958   |   0.248148   |  0.0263357   | +             |
| hendrycksTest-high_school_mathematics                     | acc_norm        |   0.251852   |  0.0264661   |   0.3        |  0.0279405   | +             |
| blimp_sentential_negation_npi_scope                       | acc             |   0.733      |  0.0139967   |   0.734      |  0.01398     |               |
| hendrycksTest-high_school_european_history                | acc             |   0.260606   |  0.0342774   |   0.278788   |  0.0350144   |               |
| hendrycksTest-high_school_european_history                | acc_norm        |   0.278788   |  0.0350144   |   0.315152   |  0.0362773   | +             |
| blimp_wh_questions_object_gap                             | acc             |   0.835      |  0.0117436   |   0.841      |  0.0115695   |               |
| hendrycksTest-international_law                           | acc             |   0.264463   |  0.0402619   |   0.214876   |  0.0374949   | -             |
| hendrycksTest-international_law                           | acc_norm        |   0.404959   |  0.0448114   |   0.438017   |  0.0452915   |               |
| hendrycksTest-high_school_us_history                      | acc             |   0.289216   |  0.0318223   |   0.323529   |  0.0328347   | +             |
| hendrycksTest-high_school_us_history                      | acc_norm        |   0.29902    |  0.0321333   |   0.323529   |  0.0328347   |               |
| openbookqa                                                | acc             |   0.29       |  0.0203132   |   0.276      |  0.0200112   |               |
| openbookqa                                                | acc_norm        |   0.382      |  0.0217508   |   0.362      |  0.0215137   |               |
| blimp_causative                                           | acc             |   0.761      |  0.013493    |   0.737      |  0.0139293   | -             |
| record                                                    | f1              |   0.885049   |  0.00314367  |   0.878443   |  0.00322394  | -             |
| record                                                    | em              |   0.8765     |  0.00329027  |   0.8702     |  0.003361    | -             |
| blimp_determiner_noun_agreement_1                         | acc             |   0.995      |  0.00223159  |   0.996      |  0.00199699  |               |
| hendrycksTest-miscellaneous                               | acc             |   0.274585   |  0.0159598   |   0.305236   |  0.0164677   | +             |
| hendrycksTest-miscellaneous                               | acc_norm        |   0.260536   |  0.015696    |   0.269476   |  0.0158662   |               |
| hendrycksTest-virology                                    | acc             |   0.349398   |  0.0371173   |   0.343373   |  0.0369658   |               |
| hendrycksTest-virology                                    | acc_norm        |   0.325301   |  0.0364717   |   0.331325   |  0.0366431   |               |
| mathqa                                                    | acc             |   0.267002   |  0.00809858  |   0.269012   |  0.00811786  |               |
| mathqa                                                    | acc_norm        |   0.270687   |  0.00813376  |   0.261642   |  0.00804614  | -             |
| squad2                                                    | exact           |  10.6207     |  0           |  10.6123     |  0           | -             |
| squad2                                                    | f1              |  17.7413     |  0           |  17.8734     |  0           | +             |
| squad2                                                    | HasAns_exact    |  15.5027     |  0           |  17.2571     |  0           | +             |
| squad2                                                    | HasAns_f1       |  29.7643     |  0           |  31.8        |  0           | +             |
| squad2                                                    | NoAns_exact     |   5.75273    |  0           |   3.98654    |  0           | -             |
| squad2                                                    | NoAns_f1        |   5.75273    |  0           |   3.98654    |  0           | -             |
| squad2                                                    | best_exact      |  50.0716     |  0           |  50.0716     |  0           |               |
| squad2                                                    | best_f1         |  50.0778     |  0           |  50.077      |  0           | -             |
| mnli_mismatched                                           | acc             |   0.376627   |  0.00488687  |   0.320586   |  0.00470696  | -             |
| blimp_animate_subject_passive                             | acc             |   0.781      |  0.0130847   |   0.79       |  0.0128867   |               |
| blimp_determiner_noun_agreement_with_adj_irregular_1      | acc             |   0.878      |  0.0103549   |   0.834      |  0.0117721   | -             |
| qnli                                                      | acc             |   0.513454   |  0.00676296  |   0.491305   |  0.00676439  | -             |
| blimp_intransitive                                        | acc             |   0.858      |  0.0110435   |   0.806      |  0.0125108   | -             |
| ethics_cm                                                 | acc             |   0.559846   |  0.00796521  |   0.512227   |  0.00802048  | -             |
| hendrycksTest-high_school_computer_science                | acc             |   0.25       |  0.0435194   |   0.2        |  0.0402015   | -             |
| hendrycksTest-high_school_computer_science                | acc_norm        |   0.27       |  0.0446196   |   0.26       |  0.0440844   |               |
| iwslt17-ar-en                                             | bleu            |  20.7322     |  0.795602    |  21.4685     |  0.64825     | +             |
| iwslt17-ar-en                                             | chrf            |   0.450919   |  0.00526515  |   0.452175   |  0.00498012  |               |
| iwslt17-ar-en                                             | ter             |   0.787631   |  0.0285488   |   0.733514   |  0.0201688   | +             |
| hendrycksTest-security_studies                            | acc             |   0.363265   |  0.0307891   |   0.391837   |  0.0312513   |               |
| hendrycksTest-security_studies                            | acc_norm        |   0.285714   |  0.0289206   |   0.285714   |  0.0289206   |               |
| hendrycksTest-global_facts                                | acc             |   0.25       |  0.0435194   |   0.29       |  0.0456048   |               |
| hendrycksTest-global_facts                                | acc_norm        |   0.22       |  0.0416333   |   0.26       |  0.0440844   |               |
| anli_r1                                                   | acc             |   0.322      |  0.0147829   |   0.297      |  0.0144568   | -             |
| blimp_left_branch_island_simple_question                  | acc             |   0.867      |  0.0107437   |   0.884      |  0.0101315   | +             |
| hendrycksTest-astronomy                                   | acc             |   0.25       |  0.0352381   |   0.25       |  0.0352381   |               |
| hendrycksTest-astronomy                                   | acc_norm        |   0.335526   |  0.038425    |   0.348684   |  0.0387814   |               |
| mrpc                                                      | acc             |   0.683824   |  0.0230483   |   0.536765   |  0.024717    | -             |
| mrpc                                                      | f1              |   0.812227   |  0.0162476   |   0.63301    |  0.0247985   | -             |
| ethics_utilitarianism                                     | acc             |   0.509775   |  0.00721024  |   0.525374   |  0.00720233  | +             |
| blimp_determiner_noun_agreement_2                         | acc             |   0.977      |  0.00474273  |   0.99       |  0.003148    | +             |
| lambada_cloze                                             | ppl             | 405.646      | 14.5519      | 388.123      | 13.1523      | +             |
| lambada_cloze                                             | acc             |   0.0199884  |  0.00194992  |   0.0116437  |  0.00149456  | -             |
| truthfulqa_mc                                             | mc1             |   0.201958   |  0.014054    |   0.225214   |  0.0146232   | +             |
| truthfulqa_mc                                             | mc2             |   0.359537   |  0.0134598   |   0.371625   |  0.0136558   |               |
| blimp_wh_vs_that_with_gap_long_distance                   | acc             |   0.342      |  0.0150087   |   0.441      |  0.0157088   | +             |
| hendrycksTest-business_ethics                             | acc             |   0.29       |  0.0456048   |   0.28       |  0.0451261   |               |
| hendrycksTest-business_ethics                             | acc_norm        |   0.3        |  0.0460566   |   0.29       |  0.0456048   |               |
| arithmetic_3ds                                            | acc             |   0.046      |  0.0046854   |   0.0065     |  0.00179736  | -             |
| blimp_determiner_noun_agreement_with_adjective_1          | acc             |   0.978      |  0.00464086  |   0.988      |  0.00344498  | +             |
| hendrycksTest-moral_disputes                              | acc             |   0.283237   |  0.0242579   |   0.277457   |  0.0241057   |               |
| hendrycksTest-moral_disputes                              | acc_norm        |   0.32659    |  0.0252483   |   0.309249   |  0.0248831   |               |
| arithmetic_2da                                            | acc             |   0.2405     |  0.00955906  |   0.0455     |  0.00466109  | -             |
| qa4mre_2011                                               | acc             |   0.458333   |  0.0456755   |   0.425      |  0.0453163   |               |
| qa4mre_2011                                               | acc_norm        |   0.533333   |  0.045733    |   0.558333   |  0.0455219   |               |
| blimp_regular_plural_subject_verb_agreement_1             | acc             |   0.968      |  0.00556839  |   0.966      |  0.00573384  |               |
| hendrycksTest-human_sexuality                             | acc             |   0.396947   |  0.0429114   |   0.389313   |  0.0427649   |               |
| hendrycksTest-human_sexuality                             | acc_norm        |   0.343511   |  0.0416498   |   0.305344   |  0.0403931   |               |
| blimp_passive_1                                           | acc             |   0.885      |  0.0100934   |   0.878      |  0.0103549   |               |
| blimp_drop_argument                                       | acc             |   0.823      |  0.0120755   |   0.784      |  0.0130197   | -             |
| hendrycksTest-high_school_microeconomics                  | acc             |   0.277311   |  0.0290794   |   0.260504   |  0.0285103   |               |
| hendrycksTest-high_school_microeconomics                  | acc_norm        |   0.39916    |  0.0318111   |   0.390756   |  0.0316938   |               |
| hendrycksTest-us_foreign_policy                           | acc             |   0.34       |  0.0476095   |   0.32       |  0.0468826   |               |
| hendrycksTest-us_foreign_policy                           | acc_norm        |   0.35       |  0.0479372   |   0.4        |  0.0492366   | +             |
| blimp_ellipsis_n_bar_1                                    | acc             |   0.841      |  0.0115695   |   0.846      |  0.0114199   |               |
| hendrycksTest-high_school_physics                         | acc             |   0.271523   |  0.0363133   |   0.264901   |  0.0360304   |               |
| hendrycksTest-high_school_physics                         | acc_norm        |   0.271523   |  0.0363133   |   0.284768   |  0.0368488   |               |
| qa4mre_2013                                               | acc             |   0.401408   |  0.0291384   |   0.362676   |  0.028579    | -             |
| qa4mre_2013                                               | acc_norm        |   0.383803   |  0.0289082   |   0.387324   |  0.0289574   |               |
| blimp_wh_vs_that_no_gap                                   | acc             |   0.969      |  0.00548353  |   0.963      |  0.00597216  | -             |
| headqa_es                                                 | acc             |   0.251276   |  0.0082848   |   0.238877   |  0.00814442  | -             |
| headqa_es                                                 | acc_norm        |   0.286652   |  0.00863721  |   0.290664   |  0.00867295  |               |
| blimp_sentential_subject_island                           | acc             |   0.421      |  0.0156206   |   0.359      |  0.0151773   | -             |
| hendrycksTest-philosophy                                  | acc             |   0.26045    |  0.0249267   |   0.241158   |  0.0242966   |               |
| hendrycksTest-philosophy                                  | acc_norm        |   0.334405   |  0.0267954   |   0.327974   |  0.0266644   |               |
| hendrycksTest-elementary_mathematics                      | acc             |   0.251323   |  0.0223405   |   0.248677   |  0.0222618   |               |
| hendrycksTest-elementary_mathematics                      | acc_norm        |   0.26455    |  0.0227175   |   0.275132   |  0.0230001   |               |
| math_geometry                                             | acc             |   0.0104384  |  0.00464863  |   0.0187891  |  0.00621042  | +             |
| blimp_wh_questions_subject_gap_long_distance              | acc             |   0.883      |  0.0101693   |   0.886      |  0.0100551   |               |
| hendrycksTest-college_physics                             | acc             |   0.205882   |  0.0402338   |   0.205882   |  0.0402338   |               |
| hendrycksTest-college_physics                             | acc_norm        |   0.245098   |  0.0428011   |   0.22549    |  0.0415831   |               |
| hellaswag                                                 | acc             |   0.49532    |  0.00498956  |   0.488747   |  0.00498852  | -             |
| hellaswag                                                 | acc_norm        |   0.66202    |  0.00472055  |   0.648277   |  0.00476532  | -             |
| hendrycksTest-logical_fallacies                           | acc             |   0.294479   |  0.0358117   |   0.269939   |  0.0348783   |               |
| hendrycksTest-logical_fallacies                           | acc_norm        |   0.355828   |  0.0376152   |   0.343558   |  0.0373113   |               |
| hendrycksTest-machine_learning                            | acc             |   0.223214   |  0.039523    |   0.339286   |  0.0449395   | +             |
| hendrycksTest-machine_learning                            | acc_norm        |   0.178571   |  0.0363521   |   0.205357   |  0.0383424   |               |
| hendrycksTest-high_school_psychology                      | acc             |   0.273394   |  0.0191093   |   0.286239   |  0.0193794   |               |
| hendrycksTest-high_school_psychology                      | acc_norm        |   0.269725   |  0.0190285   |   0.266055   |  0.018946    |               |
| prost                                                     | acc             |   0.268254   |  0.00323688  |   0.256298   |  0.00318967  | -             |
| prost                                                     | acc_norm        |   0.274658   |  0.00326093  |   0.280156   |  0.00328089  | +             |
| blimp_determiner_noun_agreement_with_adj_irregular_2      | acc             |   0.916      |  0.00877616  |   0.898      |  0.00957537  | -             |
| wnli                                                      | acc             |   0.464789   |  0.0596131   |   0.43662    |  0.0592794   |               |
| hendrycksTest-professional_law                            | acc             |   0.273794   |  0.0113886   |   0.284876   |  0.0115278   |               |
| hendrycksTest-professional_law                            | acc_norm        |   0.292699   |  0.0116209   |   0.301825   |  0.0117244   |               |
| math_algebra                                              | acc             |   0.0117944  |  0.00313487  |   0.0126369  |  0.00324352  |               |
| wikitext                                                  | word_perplexity |  10.8819     |  0           |  11.4687     |  0           | -             |
| wikitext                                                  | byte_perplexity |   1.56268    |  0           |   1.5781     |  0           | -             |
| wikitext                                                  | bits_per_byte   |   0.644019   |  0           |   0.658188   |  0           | -             |
| anagrams1                                                 | acc             |   0.0008     |  0.000282744 |   0.0125     |  0.00111108  | +             |
| math_prealgebra                                           | acc             |   0.0126292  |  0.00378589  |   0.0195178  |  0.00469003  | +             |
| blimp_principle_A_domain_2                                | acc             |   0.889      |  0.0099387   |   0.887      |  0.0100166   |               |
| cycle_letters                                             | acc             |   0.0026     |  0.000509264 |   0.0331     |  0.00178907  | +             |
| hendrycksTest-college_mathematics                         | acc             |   0.26       |  0.0440844   |   0.26       |  0.0440844   |               |
| hendrycksTest-college_mathematics                         | acc_norm        |   0.4        |  0.0492366   |   0.31       |  0.0464823   | -             |
| arithmetic_1dc                                            | acc             |   0.089      |  0.00636866  |   0.077      |  0.00596266  | -             |
| arithmetic_4da                                            | acc             |   0.007      |  0.00186474  |   0.0005     |  0.0005      | -             |
| triviaqa                                                  | acc             |   0.167418   |  0.00351031  |   0.150888   |  0.00336543  | -             |
| boolq                                                     | acc             |   0.655352   |  0.00831224  |   0.673394   |  0.00820236  | +             |
| random_insertion                                          | acc             |   0          |  0           |   0.0004     |  0.00019997  | +             |
| qa4mre_2012                                               | acc             |   0.4125     |  0.0390407   |   0.4        |  0.0388514   |               |
| qa4mre_2012                                               | acc_norm        |   0.50625    |  0.0396495   |   0.4625     |  0.0395409   | -             |
| math_asdiv                                                | acc             |   0.00563991 |  0.00156015  |   0.00997831 |  0.00207066  | +             |
| hendrycksTest-moral_scenarios                             | acc             |   0.236872   |  0.0142196   |   0.236872   |  0.0142196   |               |
| hendrycksTest-moral_scenarios                             | acc_norm        |   0.272626   |  0.0148934   |   0.272626   |  0.0148934   |               |
| hendrycksTest-high_school_geography                       | acc             |   0.20202    |  0.0286062   |   0.247475   |  0.0307463   | +             |
| hendrycksTest-high_school_geography                       | acc_norm        |   0.292929   |  0.032425    |   0.287879   |  0.0322588   |               |
| gsm8k                                                     | acc             |   0          |  0           |   0          |  0           |               |
| blimp_existential_there_object_raising                    | acc             |   0.792      |  0.0128414   |   0.812      |  0.0123616   | +             |
| blimp_superlative_quantifiers_2                           | acc             |   0.865      |  0.0108117   |   0.917      |  0.00872853  | +             |
| hendrycksTest-college_chemistry                           | acc             |   0.24       |  0.0429235   |   0.28       |  0.0451261   |               |
| hendrycksTest-college_chemistry                           | acc_norm        |   0.28       |  0.0451261   |   0.31       |  0.0464823   |               |
| blimp_existential_there_quantifiers_2                     | acc             |   0.383      |  0.0153801   |   0.545      |  0.0157551   | +             |
| hendrycksTest-abstract_algebra                            | acc             |   0.26       |  0.0440844   |   0.17       |  0.0377525   | -             |
| hendrycksTest-abstract_algebra                            | acc_norm        |   0.3        |  0.0460566   |   0.26       |  0.0440844   |               |
| hendrycksTest-professional_psychology                     | acc             |   0.28268    |  0.0182173   |   0.26634    |  0.0178832   |               |
| hendrycksTest-professional_psychology                     | acc_norm        |   0.259804   |  0.0177409   |   0.256536   |  0.0176678   |               |
| ethics_virtue                                             | acc             |   0.200201   |  0.00567376  |   0.249849   |  0.00613847  | +             |
| ethics_virtue                                             | em              |   0          |  0           |   0.0040201  |  0           | +             |
| arithmetic_5da                                            | acc             |   0.0005     |  0.0005      |   0          |  0           | -             |
| mutual                                                    | r@1             |   0.468397   |  0.0167737   |   0.455982   |  0.0167421   |               |
| mutual                                                    | r@2             |   0.735892   |  0.0148193   |   0.732506   |  0.0148796   |               |
| mutual                                                    | mrr             |   0.682186   |  0.0103375   |   0.675226   |  0.0103132   |               |
| blimp_irregular_past_participle_verbs                     | acc             |   0.876      |  0.0104275   |   0.869      |  0.0106749   |               |
| ethics_deontology                                         | acc             |   0.523637   |  0.0083298   |   0.497775   |  0.00833904  | -             |
| ethics_deontology                                         | em              |   0.0355951  |  0           |   0.00333704 |  0           | -             |
| blimp_transitive                                          | acc             |   0.855      |  0.01114     |   0.818      |  0.0122076   | -             |
| hendrycksTest-college_computer_science                    | acc             |   0.27       |  0.0446196   |   0.29       |  0.0456048   |               |
| hendrycksTest-college_computer_science                    | acc_norm        |   0.26       |  0.0440844   |   0.27       |  0.0446196   |               |
| hendrycksTest-professional_medicine                       | acc             |   0.272059   |  0.027033    |   0.283088   |  0.0273659   |               |
| hendrycksTest-professional_medicine                       | acc_norm        |   0.261029   |  0.0266793   |   0.279412   |  0.0272572   |               |
| sciq                                                      | acc             |   0.915      |  0.00882343  |   0.895      |  0.00969892  | -             |
| sciq                                                      | acc_norm        |   0.874      |  0.0104992   |   0.869      |  0.0106749   |               |
| blimp_anaphor_number_agreement                            | acc             |   0.995      |  0.00223159  |   0.993      |  0.00263779  |               |
| blimp_wh_questions_subject_gap                            | acc             |   0.913      |  0.00891687  |   0.925      |  0.00833333  | +             |
| blimp_wh_vs_that_with_gap                                 | acc             |   0.429      |  0.015659    |   0.482      |  0.015809    | +             |
| math_num_theory                                           | acc             |   0.0203704  |  0.00608466  |   0.0351852  |  0.00793611  | +             |
| blimp_complex_NP_island                                   | acc             |   0.535      |  0.0157805   |   0.538      |  0.0157735   |               |
| blimp_expletive_it_object_raising                         | acc             |   0.78       |  0.0131062   |   0.777      |  0.0131698   |               |
| lambada_mt_en                                             | ppl             |   4.10224    |  0.0884971   |   4.62504    |  0.10549     | -             |
| lambada_mt_en                                             | acc             |   0.682127   |  0.00648741  |   0.648554   |  0.00665142  | -             |
| hendrycksTest-formal_logic                                | acc             |   0.34127    |  0.042408    |   0.309524   |  0.0413491   |               |
| hendrycksTest-formal_logic                                | acc_norm        |   0.325397   |  0.041906    |   0.325397   |  0.041906    |               |
| blimp_matrix_question_npi_licensor_present                | acc             |   0.727      |  0.014095    |   0.663      |  0.0149551   | -             |
| blimp_superlative_quantifiers_1                           | acc             |   0.871      |  0.0106053   |   0.791      |  0.0128641   | -             |
| lambada_mt_de                                             | ppl             |  82.2416     |  4.88447     |  89.7905     |  5.30301     | -             |
| lambada_mt_de                                             | acc             |   0.312827   |  0.00645948  |   0.312245   |  0.0064562   |               |
| hendrycksTest-computer_security                           | acc             |   0.27       |  0.0446196   |   0.37       |  0.0485237   | +             |
| hendrycksTest-computer_security                           | acc_norm        |   0.33       |  0.0472582   |   0.37       |  0.0485237   |               |
| ethics_justice                                            | acc             |   0.526627   |  0.00960352  |   0.501479   |  0.00961712  | -             |
| ethics_justice                                            | em              |   0.0251479  |  0           |   0          |  0           | -             |
| blimp_principle_A_reconstruction                          | acc             |   0.444      |  0.0157198   |   0.296      |  0.0144427   | -             |
| blimp_existential_there_subject_raising                   | acc             |   0.875      |  0.0104635   |   0.877      |  0.0103913   |               |
| math_precalc                                              | acc             |   0.0018315  |  0.0018315   |   0.014652   |  0.00514689  | +             |
| qasper                                                    | f1_yesno        |   0.666667   |  0.0311266   |   0.632997   |  0.032868    | -             |
| qasper                                                    | f1_abstractive  |   0.118383   |  0.00692993  |   0.113489   |  0.00729073  |               |
| cb                                                        | acc             |   0.357143   |  0.0646096   |   0.196429   |  0.0535714   | -             |
| cb                                                        | f1              |   0.288109   |  0           |   0.149038   |  0           | -             |
| blimp_animate_subject_trans                               | acc             |   0.868      |  0.0107094   |   0.858      |  0.0110435   |               |
| hendrycksTest-high_school_statistics                      | acc             |   0.291667   |  0.0309987   |   0.310185   |  0.031547    |               |
| hendrycksTest-high_school_statistics                      | acc_norm        |   0.314815   |  0.0316747   |   0.361111   |  0.0327577   | +             |
| blimp_irregular_plural_subject_verb_agreement_2           | acc             |   0.919      |  0.00863212  |   0.881      |  0.0102442   | -             |
| lambada_mt_es                                             | ppl             |  83.6696     |  4.57489     |  92.1172     |  5.05064     | -             |
| lambada_mt_es                                             | acc             |   0.326994   |  0.00653569  |   0.322337   |  0.00651139  |               |
| anli_r2                                                   | acc             |   0.337      |  0.0149551   |   0.327      |  0.0148422   |               |
| hendrycksTest-nutrition                                   | acc             |   0.346405   |  0.0272456   |   0.346405   |  0.0272456   |               |
| hendrycksTest-nutrition                                   | acc_norm        |   0.401961   |  0.0280742   |   0.385621   |  0.0278707   |               |
| anli_r3                                                   | acc             |   0.3525     |  0.0137972   |   0.336667   |  0.0136476   | -             |
| blimp_regular_plural_subject_verb_agreement_2             | acc             |   0.916      |  0.00877616  |   0.897      |  0.00961683  | -             |
| blimp_tough_vs_raising_2                                  | acc             |   0.857      |  0.0110758   |   0.826      |  0.0119945   | -             |
| mnli                                                      | acc             |   0.374733   |  0.00488619  |   0.316047   |  0.00469317  | -             |
| drop                                                      | em              |   0.0228607  |  0.0015306   |   0.0595638  |  0.00242379  | +             |
| drop                                                      | f1              |   0.103871   |  0.00219977  |   0.120355   |  0.00270951  | +             |
| blimp_determiner_noun_agreement_with_adj_2                | acc             |   0.936      |  0.00774364  |   0.95       |  0.00689547  | +             |
| arithmetic_2dm                                            | acc             |   0.14       |  0.00776081  |   0.061      |  0.00535293  | -             |
| blimp_determiner_noun_agreement_irregular_2               | acc             |   0.932      |  0.00796489  |   0.93       |  0.00807249  |               |
| lambada                                                   | ppl             |   4.10224    |  0.0884971   |   4.62504    |  0.10549     | -             |
| lambada                                                   | acc             |   0.682127   |  0.00648741  |   0.648554   |  0.00665142  | -             |
| arithmetic_3da                                            | acc             |   0.0865     |  0.00628718  |   0.007      |  0.00186474  | -             |
| blimp_irregular_past_participle_adjectives                | acc             |   0.956      |  0.00648892  |   0.947      |  0.00708811  | -             |
| hendrycksTest-college_biology                             | acc             |   0.284722   |  0.0377381   |   0.201389   |  0.0335365   | -             |
| hendrycksTest-college_biology                             | acc_norm        |   0.270833   |  0.0371618   |   0.222222   |  0.0347659   | -             |
| headqa_en                                                 | acc             |   0.335522   |  0.00901875  |   0.324945   |  0.00894582  | -             |
| headqa_en                                                 | acc_norm        |   0.383297   |  0.00928648  |   0.375638   |  0.00925014  |               |
| blimp_determiner_noun_agreement_irregular_1               | acc             |   0.944      |  0.0072744   |   0.912      |  0.00896305  | -             |
| blimp_existential_there_quantifiers_1                     | acc             |   0.981      |  0.00431945  |   0.985      |  0.00384575  |               |
| blimp_inchoative                                          | acc             |   0.683      |  0.0147217   |   0.653      |  0.0150605   | -             |
| mutual_plus                                               | r@1             |   0.409707   |  0.016531    |   0.395034   |  0.0164328   |               |
| mutual_plus                                               | r@2             |   0.680587   |  0.0156728   |   0.674944   |  0.015745    |               |
| mutual_plus                                               | mrr             |   0.640801   |  0.0104141   |   0.632713   |  0.0103391   |               |
| blimp_tough_vs_raising_1                                  | acc             |   0.734      |  0.01398     |   0.736      |  0.0139463   |               |
| winogrande                                                | acc             |   0.640884   |  0.0134831   |   0.636148   |  0.0135215   |               |
| race                                                      | acc             |   0.37512    |  0.0149842   |   0.374163   |  0.0149765   |               |
| blimp_irregular_plural_subject_verb_agreement_1           | acc             |   0.918      |  0.00868052  |   0.908      |  0.00914438  | -             |
| hendrycksTest-high_school_macroeconomics                  | acc             |   0.284615   |  0.0228783   |   0.284615   |  0.0228783   |               |
| hendrycksTest-high_school_macroeconomics                  | acc_norm        |   0.276923   |  0.022688    |   0.284615   |  0.0228783   |               |
| blimp_adjunct_island                                      | acc             |   0.902      |  0.00940662  |   0.888      |  0.00997775  | -             |
| hendrycksTest-high_school_chemistry                       | acc             |   0.211823   |  0.028749    |   0.236453   |  0.0298961   |               |
| hendrycksTest-high_school_chemistry                       | acc_norm        |   0.29064    |  0.0319474   |   0.300493   |  0.032258    |               |
| arithmetic_2ds                                            | acc             |   0.218      |  0.00923475  |   0.051      |  0.00492053  | -             |
| blimp_principle_A_case_2                                  | acc             |   0.953      |  0.00669596  |   0.955      |  0.00655881  |               |
| blimp_only_npi_licensor_present                           | acc             |   0.953      |  0.00669596  |   0.926      |  0.00828206  | -             |
| math_counting_and_prob                                    | acc             |   0.0021097  |  0.0021097   |   0.0274262  |  0.00750954  | +             |
| cola                                                      | mcc             |  -0.0504508  |  0.0251594   |  -0.0854256  |  0.0304519   | -             |
| webqs                                                     | acc             |   0.0226378  |  0.00330058  |   0.023622   |  0.00336987  |               |
| arithmetic_4ds                                            | acc             |   0.0055     |  0.00165416  |   0.0005     |  0.0005      | -             |
| blimp_wh_vs_that_no_gap_long_distance                     | acc             |   0.939      |  0.00757208  |   0.94       |  0.00751375  |               |
| pile_bookcorpus2                                          | word_perplexity |  27.0559     |  0           |  28.7786     |  0           | -             |
| pile_bookcorpus2                                          | byte_perplexity |   1.78037    |  0           |   1.79969    |  0           | -             |
| pile_bookcorpus2                                          | bits_per_byte   |   0.832176   |  0           |   0.847751   |  0           | -             |
| blimp_sentential_negation_npi_licensor_present            | acc             |   0.982      |  0.00420639  |   0.994      |  0.00244335  | +             |
| hendrycksTest-high_school_government_and_politics         | acc             |   0.227979   |  0.0302769   |   0.274611   |  0.0322102   | +             |
| hendrycksTest-high_school_government_and_politics         | acc_norm        |   0.248705   |  0.0311958   |   0.259067   |  0.0316188   |               |
| blimp_ellipsis_n_bar_2                                    | acc             |   0.916      |  0.00877616  |   0.937      |  0.00768701  | +             |
| hendrycksTest-clinical_knowledge                          | acc             |   0.267925   |  0.0272573   |   0.283019   |  0.0277242   |               |
| hendrycksTest-clinical_knowledge                          | acc_norm        |   0.316981   |  0.0286372   |   0.343396   |  0.0292245   |               |
| mc_taco                                                   | em              |   0.132883   |  0           |   0.125375   |  0           | -             |
| mc_taco                                                   | f1              |   0.499712   |  0           |   0.487131   |  0           | -             |
| wsc                                                       | acc             |   0.365385   |  0.0474473   |   0.365385   |  0.0474473   |               |
| hendrycksTest-college_medicine                            | acc             |   0.190751   |  0.0299579   |   0.231214   |  0.0321474   | +             |
| hendrycksTest-college_medicine                            | acc_norm        |   0.265896   |  0.0336876   |   0.289017   |  0.0345643   |               |
| hendrycksTest-high_school_world_history                   | acc             |   0.2827     |  0.0293128   |   0.295359   |  0.0296963   |               |
| hendrycksTest-high_school_world_history                   | acc_norm        |   0.312236   |  0.0301651   |   0.312236   |  0.0301651   |               |
| hendrycksTest-anatomy                                     | acc             |   0.281481   |  0.03885     |   0.296296   |  0.0394462   |               |
| hendrycksTest-anatomy                                     | acc_norm        |   0.266667   |  0.0382017   |   0.288889   |  0.0391545   |               |
| hendrycksTest-jurisprudence                               | acc             |   0.277778   |  0.0433004   |   0.25       |  0.0418609   |               |
| hendrycksTest-jurisprudence                               | acc_norm        |   0.425926   |  0.0478034   |   0.416667   |  0.0476608   |               |
| logiqa                                                    | acc             |   0.211982   |  0.016031    |   0.193548   |  0.0154963   | -             |
| logiqa                                                    | acc_norm        |   0.291859   |  0.0178316   |   0.281106   |  0.0176324   |               |
| ethics_utilitarianism_original                            | acc             |   0.941556   |  0.00338343  |   0.767679   |  0.00609112  | -             |
| blimp_principle_A_c_command                               | acc             |   0.81       |  0.0124119   |   0.827      |  0.0119672   | +             |
| blimp_coordinate_structure_constraint_complex_left_branch | acc             |   0.764      |  0.0134345   |   0.794      |  0.0127956   | +             |
| arithmetic_5ds                                            | acc             |   0          |  0           |   0          |  0           |               |
| lambada_mt_it                                             | ppl             |  86.66       |  5.1869      |  96.8846     |  5.80902     | -             |
| lambada_mt_it                                             | acc             |   0.336891   |  0.0065849   |   0.328158   |  0.00654165  | -             |
| wsc273                                                    | acc             |   0.827839   |  0.0228905   |   0.827839   |  0.0228905   |               |
| blimp_coordinate_structure_constraint_object_extraction   | acc             |   0.876      |  0.0104275   |   0.852      |  0.0112349   | -             |
| blimp_principle_A_domain_3                                | acc             |   0.819      |  0.0121814   |   0.79       |  0.0128867   | -             |
| blimp_left_branch_island_echo_question                    | acc             |   0.519      |  0.0158079   |   0.638      |  0.0152048   | +             |
| rte                                                       | acc             |   0.548736   |  0.0299531   |   0.534296   |  0.0300256   |               |
| blimp_passive_2                                           | acc             |   0.899      |  0.00953362  |   0.892      |  0.00982     |               |
| hendrycksTest-electrical_engineering                      | acc             |   0.358621   |  0.0399663   |   0.344828   |  0.0396093   |               |
| hendrycksTest-electrical_engineering                      | acc_norm        |   0.372414   |  0.0402873   |   0.372414   |  0.0402873   |               |
| sst                                                       | acc             |   0.493119   |  0.0169402   |   0.626147   |  0.0163938   | +             |
| blimp_npi_present_1                                       | acc             |   0.576      |  0.0156355   |   0.565      |  0.0156851   |               |
| piqa                                                      | acc             |   0.754081   |  0.0100473   |   0.739391   |  0.0102418   | -             |
| piqa                                                      | acc_norm        |   0.761697   |  0.00994033  |   0.755169   |  0.0100323   |               |
| hendrycksTest-professional_accounting                     | acc             |   0.265957   |  0.0263581   |   0.312057   |  0.0276401   | +             |
| hendrycksTest-professional_accounting                     | acc_norm        |   0.22695    |  0.0249871   |   0.27305    |  0.0265779   | +             |
| arc_challenge                                             | acc             |   0.337884   |  0.013822    |   0.325085   |  0.0136881   |               |
| arc_challenge                                             | acc_norm        |   0.366041   |  0.0140772   |   0.352389   |  0.0139601   |               |
| hendrycksTest-econometrics                                | acc             |   0.245614   |  0.0404934   |   0.263158   |  0.0414244   |               |
| hendrycksTest-econometrics                                | acc_norm        |   0.27193    |  0.0418577   |   0.254386   |  0.0409699   |               |
| headqa                                                    | acc             |   0.251276   |  0.0082848   |   0.238877   |  0.00814442  | -             |
| headqa                                                    | acc_norm        |   0.286652   |  0.00863721  |   0.290664   |  0.00867295  |               |
| wic                                                       | acc             |   0.5        |  0.0198107   |   0.482759   |  0.0197989   |               |
| hendrycksTest-high_school_biology                         | acc             |   0.251613   |  0.024686    |   0.270968   |  0.0252844   |               |
| hendrycksTest-high_school_biology                         | acc_norm        |   0.283871   |  0.0256494   |   0.274194   |  0.0253781   |               |
| hendrycksTest-management                                  | acc             |   0.23301    |  0.0418583   |   0.281553   |  0.0445325   | +             |
| hendrycksTest-management                                  | acc_norm        |   0.320388   |  0.0462028   |   0.291262   |  0.0449868   |               |
| blimp_npi_present_2                                       | acc             |   0.664      |  0.0149441   |   0.645      |  0.0151395   | -             |
| hendrycksTest-prehistory                                  | acc             |   0.243827   |  0.0238919   |   0.265432   |  0.0245692   |               |
| hendrycksTest-prehistory                                  | acc_norm        |   0.219136   |  0.0230167   |   0.225309   |  0.0232462   |               |
| hendrycksTest-world_religions                             | acc             |   0.333333   |  0.0361551   |   0.321637   |  0.0358253   |               |
| hendrycksTest-world_religions                             | acc_norm        |   0.380117   |  0.0372297   |   0.397661   |  0.0375364   |               |
| math_intermediate_algebra                                 | acc             |   0.00332226 |  0.00191598  |   0.00996678 |  0.00330749  | +             |
| anagrams2                                                 | acc             |   0.0055     |  0.000739615 |   0.0347     |  0.00183028  | +             |
| arc_easy                                                  | acc             |   0.669613   |  0.00965143  |   0.647306   |  0.00980442  | -             |
| arc_easy                                                  | acc_norm        |   0.622896   |  0.00994504  |   0.609848   |  0.0100091   | -             |
| blimp_anaphor_gender_agreement                            | acc             |   0.994      |  0.00244335  |   0.993      |  0.00263779  |               |
| hendrycksTest-marketing                                   | acc             |   0.307692   |  0.0302364   |   0.311966   |  0.0303515   |               |
| hendrycksTest-marketing                                   | acc_norm        |   0.294872   |  0.0298726   |   0.34188    |  0.031075    | +             |
| blimp_principle_A_domain_1                                | acc             |   0.997      |  0.00173032  |   0.997      |  0.00173032  |               |
| blimp_wh_island                                           | acc             |   0.852      |  0.0112349   |   0.856      |  0.011108    |               |
| hendrycksTest-sociology                                   | acc             |   0.278607   |  0.0317006   |   0.303483   |  0.0325101   |               |
| hendrycksTest-sociology                                   | acc_norm        |   0.318408   |  0.0329412   |   0.298507   |  0.0323574   |               |
| blimp_distractor_agreement_relative_clause                | acc             |   0.719      |  0.0142212   |   0.774      |  0.0132325   | +             |
| truthfulqa_gen                                            | bleurt_max      |  -0.814228   |  0.0172128   |  -0.811655   |  0.0180743   |               |
| truthfulqa_gen                                            | bleurt_acc      |   0.329253   |  0.0164513   |   0.395349   |  0.0171158   | +             |
| truthfulqa_gen                                            | bleurt_diff     |  -0.185905   |  0.0169617   |  -0.0488385  |  0.0204525   | +             |
| truthfulqa_gen                                            | bleu_max        |  20.2238     |  0.711772    |  20.8747     |  0.717003    |               |
| truthfulqa_gen                                            | bleu_acc        |   0.281518   |  0.015744    |   0.330477   |  0.0164668   | +             |
| truthfulqa_gen                                            | bleu_diff       |  -6.66121    |  0.719366    |  -2.12856    |  0.832693    | +             |
| truthfulqa_gen                                            | rouge1_max      |  45.3457     |  0.89238     |  47.0293     |  0.962404    | +             |
| truthfulqa_gen                                            | rouge1_acc      |   0.257038   |  0.0152981   |   0.341493   |  0.0166007   | +             |
| truthfulqa_gen                                            | rouge1_diff     | -10.1049     |  0.8922      |  -2.29454    |  1.2086      | +             |
| truthfulqa_gen                                            | rouge2_max      |  28.7438     |  0.981282    |  31.0617     |  1.08725     | +             |
| truthfulqa_gen                                            | rouge2_acc      |   0.201958   |  0.014054    |   0.247246   |  0.0151024   | +             |
| truthfulqa_gen                                            | rouge2_diff     | -11.0916     |  1.01664     |  -2.84021    |  1.28749     | +             |
| truthfulqa_gen                                            | rougeL_max      |  42.6116     |  0.893252    |  44.6463     |  0.966119    | +             |
| truthfulqa_gen                                            | rougeL_acc      |   0.24235    |  0.0150007   |   0.334149   |  0.0165125   | +             |
| truthfulqa_gen                                            | rougeL_diff     | -10.4299     |  0.904205    |  -2.50853    |  1.22016     | +             |
| hendrycksTest-public_relations                            | acc             |   0.281818   |  0.0430912   |   0.3        |  0.0438931   |               |
| hendrycksTest-public_relations                            | acc_norm        |   0.163636   |  0.0354343   |   0.190909   |  0.0376443   |               |
| blimp_distractor_agreement_relational_noun                | acc             |   0.833      |  0.0118004   |   0.859      |  0.0110109   | +             |
| lambada_mt_fr                                             | ppl             |  51.7313     |  2.90272     |  57.0379     |  3.15719     | -             |
| lambada_mt_fr                                             | acc             |   0.40947    |  0.00685084  |   0.388512   |  0.0067906   | -             |
| blimp_principle_A_case_1                                  | acc             |   1          |  0           |   1          |  0           |               |
| hendrycksTest-medical_genetics                            | acc             |   0.31       |  0.0464823   |   0.37       |  0.0485237   | +             |
| hendrycksTest-medical_genetics                            | acc_norm        |   0.39       |  0.0490207   |   0.41       |  0.0494311   |               |
| qqp                                                       | acc             |   0.383626   |  0.00241841  |   0.364383   |  0.00239348  | -             |
| qqp                                                       | f1              |   0.451222   |  0.00289696  |   0.516391   |  0.00263674  | +             |
| iwslt17-en-ar                                             | bleu            |   4.98225    |  0.275369    |   2.35563    |  0.188638    | -             |
| iwslt17-en-ar                                             | chrf            |   0.277708   |  0.00415432  |   0.140912   |  0.00503101  | -             |
| iwslt17-en-ar                                             | ter             |   0.954701   |  0.0126737   |   1.0909     |  0.0122111   | -             |
| multirc                                                   | acc             |   0.0178384  |  0.00428994  |   0.0409234  |  0.00642087  | +             |
| hendrycksTest-human_aging                                 | acc             |   0.264574   |  0.0296051   |   0.264574   |  0.0296051   |               |
| hendrycksTest-human_aging                                 | acc_norm        |   0.237668   |  0.0285681   |   0.197309   |  0.0267099   | -             |
| reversed_words                                            | acc             |   0          |  0           |   0.0003     |  0.000173188 | +             |
<figcaption><p>Some results are missing due to errors or computational constraints.</p>
</figcaption></figure>


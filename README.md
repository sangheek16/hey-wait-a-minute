---
output:
  html_document: default
  pdf_document: default
---

# DGRC: Divide, Generate, Recombine, and Compare

This repository contains the full experimental pipeline for the **DGRC** framework, reported in "[Hey, wait a minute: on at-issue sensitivity in Language Models](https://doi.org/10.48550/arXiv.2510.12740)" by [Sanghee J. Kim](https://sangheekim.net) and [Kanishka Misra](https://kanishka.website).

DGRC is a method for evaluating dialogue naturalness that, while retaining surprisal-based minimal-pair evaluation, offers the flexibility of open-ended continuation. We leverage a linguistic notion of <i>at-issueness</i>. DGRC happens in 4 stages: (i) **Divide** a dialogue as a prompt, (ii) **Generate** continuations for subparts using LMs, (iii) **Recombine** the dialogue and continuations, and (iv) **Compare** the likelihoods of the recombined sequences. We suggest that this approach mitigates bias in linguistic analyses of LMs and enables systematic testing of discourse-sensitive behavior.

<p align="center">
  <img src="data/figures/dgrc-4steps.pdf" alt="DGRC overview figure" width="650"/>
</p>

**Figure 1.** Visualization of the DGRC method, involving four steps:  
1) Dividing the original utterance into sub-utterances;  
2) Generating continuations for individual sub-utterances;  
3) Recombining the sub-utterances into the original utterance; and  
4) Comparing likelihoods for generated continuations.  
The end result of this process allows us to characterize an LM’s dialogue-response dynamics.

---

## Repository Structure

```
.
├── data/
│   ├── kim22_used_items.csv                  # Source stimuli (from Kim et al. 2022)
│   ├── stimuli/
│   │   ├── kim22-arc-unique.csv              # ARC stimuli
│   │   └── kim22-coord-unique.csv            # COORD stimuli
│   ├── generations/
│   │   └── <model_name>/                     # Model generations (empty folder)
│   ├── results/
│   │   ├── sorted-generations/
│   │   │   └── {freeform,rejection}/{model_name}-{arc,coord}.csv
│   │   └── dgrc/
│   │       └── {freeform,rejection}-{arc,coord}/{model_name}.csv
│   └── figures/
│       └── dgrc-4steps.pdf                   # DGRC figure (shown above)
│
├── src/
│   ├── kim22-dcpmi-stimuli.py                # Step 0: prepare data (split Kim et al. 2022 dataset)
│   ├── collect-generations.py                # Steps 1–2: divide and generate
│   ├── coalesce-generations.py               # Step 3: recombine
│   ├── dgrc-eval.py                          # Step 4: main DGRC evaluation
│   ├── dgrc-rejection-eval.py                # Step 4: rejection evaluation
│   └── utils.py                              # Shared helper functions
│
├── scripts/
│   ├── collect-generations.sh                # Wrapper for Steps 1–2
│   ├── collect-generations-model.sh          # For instruct-tuned models
│   ├── collect-generations-non-instruct.sh   # For base/non-instruct models
│   └── dgrc.sh                               # Wrapper for Step 4
│
├── requirements.txt                          
└── README.md

```

## Workflow
### Prepare Data
This code imports the dataset used in [Kim et al. (2022)](https://aclanthology.org/2022.coling-1.72/) and splits it into two datasets used for the ARC and COORD conditions.
<ul>
  <li> Input: `data/kim22_used_items.csv`
  <li> Script: `src/kim22-dcpmi-stimuli.py`
  <li> Output: `data/stimuli/kim22-arc-unique.csv` and `data/stimuli/kim22-coord-unique.csv`
</ul>

### Steps 1 & 2: Divide and Generate
Divide the original utterance into sub-utterances and Generate model continuations for each.
<ul>
  <li> Input: `data/kim22_used_items.csv`
  <li> Script: Run `scripts/collect-generations.sh` to exectue both `scripts/collect-generations-model.sh` (for instruct-tuned models) and `scripts/collect-generations-non-instruct.sh` (for base/non-instruct models). Both scripts use `src/collect_generations.py` internally.
  <li> Output: `data/generations/{model_name}/` This folder is empty in this repository, but will created when the code is executed.
</ul>

### Step 3: Recombine
Recombine the generations that were cretaed based on sub-utterances with the original utterance.
<ul>
  <li> Input: `data/stimuli/kim22-arc-unique.csv` and `data/stimuli/kim22-coord-unique.csv`
  <li> Script: Run `src/coalesce-generations.py`
  <li> Output: `data/results/sorted-generations/{freeform,rejection}/{model_name}-{arc,coord}.csv`
</ul>

### Step 4: Compare Surprisals
Compare likelihoods for the recombined generated continuations.
<ul>
  <li> Input: `data/results/sorted-generations/{freeform,rejection}/{model_name}-{arc,coord}.csv`
  <li> Script: Use `scripts/dgrc.sh` to execute both `src/dgrc-eval.py` and `src/dgrc-rejection-eval.py`
  <li> Output: `data/results/dgrc/{freeform,rejection}-{arc,coord}/{model_name}.csv`
</ul>

## Dependencies

Install all dependencies with: 
```pip install -r requirements.txt```

They include:
```
minicons
torch
transformers
```

# How to cite
```
@inproceedings{kim2025atissue,
  title = {Hey, wait a minute: on at-issue sensitivity in Language Models},
  author = {Kim, Sanghee J. and Kanishka Misra},
  year = {2025},
  url = "https://doi.org/10.48550/arXiv.2510.12740"
}
```
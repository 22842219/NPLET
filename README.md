# NPLET

This repository demonstrates model called **NPLET: A Neural Probabilistic Logical Model for Entity Typing.**

## **Model**

The framework explicitly  represents  types’  hierarchical  information  in  a proposition manner and allows the introduction of common-sense knowledge (i.e., mutually exclusive constraint and containment constraint). Formulated as a Bayesian belief update process, the output of an entity typing neural network is conditioned by a logic theory and compiled into Sentential Decision Diagram (SDD) for computation efficiency.

The framework consists of two main parts, neural network model **(NN)**, SDD-based Bayesian inference module **(SBI)** to encode hierarchical relationship among labels as a regularization term to constrain BCE loss function.

## 1. **NN**

Neural network model use **[LUKE](https://github.com/studio-ousia/luke)** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) to get input embedding vectors, and feedforward network which is a typing score layer.

| Shape  | LUKE-500K (large)                   | Typing score layer             |
| ------ | ----------------------------------- | ------------------------------ |
| Input  | [batch\_size, max\_mention\_length] | [batch\_size, 1024]            |
| Output | [batch\_size, 1024]                 | [batch\_size, num\_of\_labels] |

## 2. **SBI**

<!-- **Problem formulation**

 - Graphical model
 - Approximate inference? -->

By building a diagram of target variables, with each node in the diagram reporting whether the ... are satisfied. The leaves of this diagram are the variables for each clause. The root of the diagram reports if the entire problem is satisfied. The marginal distribution over the root of the diagram specifies what fraction of assignments satisfy the problem.

The idea  is to  update the  logistic probability  which is learned by a neural network model after the relevant evidence or background knowledge is taken into account.

This problem is interpreted to compute a **success probability** of  <img src="https://latex.codecogs.com/svg.image?y" title="y" />, namely the probability of a ground fact, given <img src="https://latex.codecogs.com/svg.image?P(\hat&space;y)" title="P(\hat y)" />. It is equal to the weighted model count (WMC) of the worlds where this query is true, i.e.,


<img src="https://latex.codecogs.com/svg.image?P(y|\hat&space;y;\alpha)&space;=&space;\frac{WMC(T\wedge&space;y)}{WMC(T)}" title="P(y|\hat y;\alpha) = \frac{WMC(T\wedge y)}{WMC(T)}" />

<img src="https://latex.codecogs.com/svg.image?WMC(T)&space;=&space;\sum_{m&space;\in&space;M(T)}&space;\prod_{l&space;\in&space;m}f(l)" title="WMC(T) = \sum_{m \in M(T)} \prod_{l \in m}f(l)" />

<img src="https://latex.codecogs.com/svg.image?WMC(T&space;\wedge&space;y)&space;=&space;\sum_{m&space;\in&space;M(T)}&space;\prod_{l&space;\in&space;m&space;}f(l|y)" title="WMC(T \wedge y) = \sum_{m \in M(T)} \prod_{l \in m }f(l|y)" />


Where,

- <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" /> is the parameters in the logical program.
- <img src="https://latex.codecogs.com/svg.image?T" title="T" /> is a propositional logic theroy over a set of observed target variables <img src="https://latex.codecogs.com/svg.image?Y&space;=&space;\{y_1,&space;...,&space;y_n\}" title="Y = \{y_1, ..., y_n\}" />
- <img src="https://latex.codecogs.com/svg.image?f" title="f" /> is a labeling function, namely <img src="https://latex.codecogs.com/svg.image?L&space;->&space;WMC" title="L -> WMC" />, mapping literals L to the variables of Y, associated with neural predicted probability <img src="https://latex.codecogs.com/svg.image?P(\hat&space;y)" title="P(\hat y)" />.

## **Datasets**

The statistics of BBN\_Modified the model has been tested are shown below:

| Dataset             | Train  | Dev    | Test  | # Types | # Levels of types' hierarchy |
| ------------------- | ------ | ------ | ----- | ------- | ---------------------------- |
| bbn_modified        | 5,143  | 644    | 644   | 48      | 2                            |
| ontonotes\_modified | 1,048  | 132    | 132   | 72      | 3                            |
| bbn                 | 29,466 | 3,273  | 6,431 | 56      | 2                            |
| ontonotes           | 79,456 | 88,284 | 1,312 | 92      | 3                            |

## **Experiments**

### 1. Baselines

This is the report of one fully-connected layer on top of luke embedding model

| Dataset            | Micro F1 | Macro F1 | Strict Acc |
| ------------------ | -------- | -------- | ---------- |
| bbn_modified       | 0.8857   | 0.8771   | 0.8197     |
| ontonotes_modified |          |          |            |


### 2. NN+SDD

| Dataset            | Micro F1 | Macro F1 | Strict Acc |
| ------------------ | -------- | -------- | ---------- |
|                    |          |          |            |
| ontonotes_modified |          |          |            |

## **How to run the model?**

### 1. Install packages

- Set up the virtual environment

  ```
    python -m virtualenv venv
    source venv/bin/activate
  ```
- The main requirements are:

  - ``pip install -r requirements.txt``
  - Install [pysdd](https://github.com/wannesm/PySDD)
  - Download [LUKE](https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing) and put it under NPLET folder

### 2. Config model

- Baseline model

We set argument ``is_sdd`` as ``False`` in instantiated class ``EntityTyping`` to disable SBI module.

- NN+SBI module

We set argument ``is_sdd`` as ``True`` in instantiated class ``EntityTyping``.

To compute semantic loss, we firstly construct SDD via ``symbolic.py`` locating under ``src/entity_typing``.

```
cd src
python symbolic.py --dataset 'ontonotes_modified' --label_size 72
```

Then we go to main.py and set ``is_sdd`` as ``True``.

### 3. Train the model

```
python -m cli \
    --model-file=luke_large_500k.tar.gz \
    entity-typing run \
    --train-batch-size=2 \
    --gradient-accumulation-steps=2 \
    --learning-rate=1e-5 \
    --num-train-epochs=22
```

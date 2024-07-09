# Project : Automatic Documentation Generation

Welcome to the IRIS Labs Recruitment Task! This year at IRIS Labs, we are focussing on **Automating Code Documentation**, with the help of LLMs and other NLP methodologies!

## Motivation 

Maintaining up-to-date and accurate documentation is important for the success and sustainability of projects. However, the manual creation and upkeep of documentation are often time-consuming and prone to errors. This project proposes the development and implementation of an automated documentation generation system, to streamline and enhance the documentation process, ensuring consistency, accuracy, and efficiency.

## General Instructions

This repository contains the code for a project we had undertaken, with a similar idea. Please go through the codebase thoroughly.

For the purpose of the recruitment task, we are listing a few issues below you can tackle. You can choose any issue(s), and work on them. Before going through any of the issues, it is suggested you go through the [README.md](README.md) first.

This will help you understand the issues below better.

|Issue | Level| 
| :---:   | :---: |
| [Dendrogram Feature Addition](#dendrogram-feature-addition)|Easy |
|  |  |
|[Documentation Customization](#documentation-customization)| Easy|
|  |  |
|[Additional Code Features](#additional-code-features)| Easy|
|  |  |
|[Knowledge Graphs Addition](#knowledge-graphs-addition) |Medium |
|  |  |
|[Investigating Clustering](#investigating-clustering) |Medium |
|  |  |
| [Investigating Embeddings Mechanisms](#investigating-embedding-mechanisms) | Hard|
|  |  |
| [Investigating Context Specific Mechanisms](#investigating-context-specific-mechanisms) | Hard|

## Task Details

### Dendrogram Feature Addition

Based on the current clustering algorithm used, contributors can add this functionality. You can look into dendrograms here. This is a very easy issue to fix.

### Documentation Customization

Customization should adhere to the needs of the user. Contributors can look into prompt engineering as the starting point, and expand from there. An example of customization should be a specific format dictated by the user.

### Additional Code Features

We have added :

1) Code Refactoring
2) Automatic Test Generation
as of now.

Various other features regarding code analytics can be added. (For example, number of functions, classes etc, which can be easily done during prompting)

### Knowledge Graphs Addition

Contributors can look into incorporating knowledge graphs based on the clustered or non-clustered files to provide further insight in the structure of the code. Retrieval of code can also be a subsequent feature after this.

### Investigating Clustering

Agglomerative Clustering was chosen to take an advantage of any hierarchical relation between the code clusters, but other more efficient or more meaningful clustering algorithms can be looked into.

### Investigating Embedding Mechanisms

We are currently using CodeBERT, which has its own limitations. Contributors can look into exploring different embedding models, perhaps even look into training their own from a multimodal approach (NL/PL).

### Investigating Context Specific Mechanisms

We are currently using a "window" algorithm to maintain context, which is not the most efficient technique. Contributors can look into memory networks, various attention mechanisms, or Reinforcement Learning with Memory.



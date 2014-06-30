---
layout: global
title: Topic Modeling - MLlib
displayTitle: <a href="mllib-guide.html">MLlib</a> - Clustering
---

* Table of contents
{:toc}

## Topic Modeling

Topic modeling is an application of machine learning to text analysis. Topic modeling is useful for different text analysis tasks, for example: document categorization, spam detection, phishing detection and many other applications.

One of the widespread algorithms is Probabilistic Latent Semantic Analysis (PLSA), suggested by Thomas Hofmann in 1999.

In the year 2013 Vorontsov and Potapenko suggested [1] an extention of PLSA model -- Robust PLSA model that is able to extract noise and backgound terms. A year later the same authours suggested to consider a Robust PLSA [2]  

##  Probabilistic Latent Semantic Analysis

PLSA is based on generative model ”bag of words”: every document is assumed to be a multinomial distribution over topics. Every topic is a multinomial distribution over words. Generative model may be defined as follows:
* For every position in document *d* i.i.d choose topic *t* from distribution of topics by document
* Choose word *w* from topic *t*

The aim of topic modeling is to recover topics and distributions of documents by topics. That may be done through solving an optimization problem, but this omtimization problem has multiple solution. That's why regularization may be necessary. It can help to 

* extract sparse topics
* extract human interpretable topics
* perform semi-supervised training
* sort out non-topic specific terms

Also a well known Latent Dirichlet Allocation model (a de-facto standart) is just a PLSA with Dirichlet Regularization. 

## Robust PLSA
The only difference between Robust PLSA and PLSA is that a word may be generated from the noise or from the background. 
 Noise is a multinumial distribution over words specific for every document and background is a multinomial distribution over words that's the same for all the documents in the collection given. 

Unfortunately, one cannot infer probabilities for a word to be generated from noise and background, so we have to bring a pair of hyperparameters -- *gamma* and *epsilon* and suppose that the probability for a word to be generated from the noise equals to   
*epsilon / (1+epsilon + gamma)* ans the probability for a word to be generated from the background equals to   *gamma / (1+epsilon + gamma)*

## Examples

*PLSASuite* and *RobustPLSASuite* are a good example of PLSA object instantiation and their superclass *AbstractTopicModelSuite* is a good example of PLSA training. 

## References

[1] Potapenko, K. Vorontsov. 2013. Robust PLSA performs better than LDA. In Proceedings of ECIR'13.

[2] Vorontsov, Potapenko. Tutorial on Probabilistic Topic Modeling: Additive Regularization for Stochastic Matrix Factorization.

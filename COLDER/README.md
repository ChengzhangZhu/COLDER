# COLDER package

## Introduction

COLDER package is a tool for spam review detection with cold-start problem.
This package is wrote by `python` with `Tensorflow`.

## Major Files and Functions

+ `context_generator.py`: social and behavior context generation
+ `graph.py`: building graph from social network
+ `sample_generator.py`: training sample generation via negative sampling
+ `train.py`: training
+ `prediction.py`: prediction

## Dependent Packages
+ tensorflow
+ numpy
+ pandas
+ re
+ tqdm
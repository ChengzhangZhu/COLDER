# COLDER package

## Introduction

COLDER package is a tool for spam review detection with cold-start problem.
This package is wrote by `python` with `keras` and Tensorflow`.

## Major Files and Functions

+ `context_generator.py`: social and behavior context generation
+ `graph.py`: building graph from social network
+ `sample_generator.py`: training sample generation via negative sampling
+ `train.py`: training
+ `prediction.py`: prediction
+ `COLDER.py`: define the COLDER architecture

## Dependent Packages
+ tensorflow
+ keras
+ numpy
+ pandas
+ re
+ tqdm

## Demo

+ Generate data --> Training model --> Testing model

  `python train.py --generate_data Y --training Y --trn_begin_date 2004-10-20 --trn_end_date 2006-10-20 --tst_begin_date 2006-10-21 --tst_end_date 2007-10-21 --epochs 3 --testing Y`

+ Training model --> Testing model

  `python train.py --training Y --trn_begin_date 2004-10-20 --trn_end_date 2006-10-20 --tst_begin_date 2006-10-21 --tst_end_date 2007-10-21 --epochs 3 --testing Y`

+ Testing model

  `python train.py --trn_begin_date 2004-10-20 --trn_end_date 2006-10-20 --tst_begin_date 2006-10-21 --tst_end_date 2007-10-21 --testing Y`
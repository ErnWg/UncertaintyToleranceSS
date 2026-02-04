# Uncertainty Tolerance Dissociates Sensation Seeking and Impulsivity

This repository contains all data and code to reproduce the results of Experiment 2 for the paper:

Wong, E., Hauser, T. U., Chandrasekaran, A., Pietrini, P., & Wu, C. M. (2026, January 28). Uncertainty Tolerance Dissociates Sensation Seeking and Impulsivity. Retrieved from osf.io/preprints/psyarxiv/yq8ce_v1

Data for Experiment 1 is not publicly availabel due to ethical guidelines.


## Software requirements

The analysis code was written in R 4.4.1. Statistical models are fit using the Stan MCMC engine via the rstan (2.32.6), which require a C++ compiler (installation instructions are available at https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started and https://mc-stan.org/cmdstanr/articles/cmdstanr.html).

## Directory structure


* `PaperFigures/`: contains all main and supplementary figures in the manuscript
* `STANmodels/`: contains code for running computational models
* `data/`: contains raw and preprocessed experiment data including behavioural and questionnaire data
* `data/recoveryAnlysis/simulations`: contains simulated data generated from `modelSimulations.R`
* `data/recoveryAnlysis/recovery`: contains results for parameter recovery from `paramRecovery.R`
* `results/`: contains results from computational modelling of experimental data

## Analyses

This folder contains participant data and the code used to perform behavioral analyses

The data folders are:
* `data/choiceDataRaw.csv/.rds` contains all the participant task data.
* `data/questionnaireScoresRaw.csv/.rds` contains all the participant questionniare data.
* `data/questionnaireSubScores.csv/.rds` contains all the participant questionniare subscore data.
* `data/stanData.RData` contains the input for STAN model fitting in `fitModels.R`

These data files are formatted using `dataFilterChecks.R` 

The main analyses reported in the paper are based on:
* `analysisMain_Final.Rmd` contains all analyses to replicate Experiment 2 figures and statistical analysis reported in the paper.

## Modeling

This folder contains the code used to perform computational modeling

* `STANmodels/` contains the various computational models implemented in STAN. 
* `fitModels.R` code to fit experimental data to the models in `STANmodels/`
* `BLPreds.R` code to generate model predictions that are used for Figure 4A

## Simulations
Simulations of task behaviour based on computational models.

* `modelSimulations.R` contains code to simulate behaviour under different compuational models 
* `fitSims.R` code to fit simulated data to the models in `STANmodels/`

### Usage to replicate figures and statistical analyses
Clone the repository and run analysisMain_Final.Rmd
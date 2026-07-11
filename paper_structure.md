# Topics to be included in the paper

Here we discuss how to elaborate the final paper

1. Intro

    1. physical and aplication contexts: introduce briefly the work done in Westfal et al. including modelling of the thermal decay 

    1. why classical approaches are limited: talk about over-reliance in restrictions on the domain of the data and low feasibility of uncertanty quantification on the frequentist approach

    1. why traditional bayesian inference won't work: talk about the computational time required to run the inference live on a real application

    1. proposed solution using NPE-PFNs


1. Prior definition

    1. dynamics of the newton law of cooling: talk about the trivial heat dissipation hipoteses

    1. aquisition of intervals on the parameter: use the collected data in order to estimate intervals for the parameters used in the simulations

    1. synthetic data generation: describe the simulator that generates synthetic data, as well as boundaries for the coefficient values

1. architecture of the PFN and NPE-PFN

    1. overview of the architectures

    1. talk about training mechanism and how we dont need real-world data

1. Training

    1. Validation on real world data

    1. Loss formulation

    1. Training curves

1. Experiments and results

    1. Validation of the posteriors and predictive posterior using the real data

    1. evaluate the uncertanty change under data scarcity, locally and globally

1. Discussion and future work

    1. Is it better than the classical, simpler alternative?

    1. Using this framework for more complex thermal regimes, that are now well modeled with newtons cooling law




When we have this, we can do:

1. Comparison with another models

    1. normalizing flows    

        1. Neural Spline Flows (NSF)

        1. Autorregressive Flows (MAF)

    1. QIRT work
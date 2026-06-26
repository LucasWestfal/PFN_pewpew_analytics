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

    1. synthetic data generation: describe the simulator that generates synthetic data

1. architecture of the PFN

    1. encoding

    1. inference

        1. using gaussian mixtures, for a unimodal decay hipoteses

        1. using normalizing flow, for generic decays with multiple discharges (needs more data! maybe it is too much)


1. Training

    1. Loss formulation

    1. Training curves

1. Experiments and results

    1.  Validation of the posteriors and predictive posterior using the real data
    1. evaluate the uncertanty change under data scarcity, locally and globally

1. Discussion and future work

    1. Is it better than the classical, simpler alternative?
    1. Using this framework for more complex thermal regimes, that are now well modeled with newtons cooling law

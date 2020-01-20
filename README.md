# Modelling SIRS infection with mobility and network structure
## Life Data Epidemiology - Nicola Dainese, Clara Eminente

This repository contains the report, the code and the data produced for the course of Life Data Epidemiology attended at University of Padua during accademic year 2019/2020.

**Abstract:**
*In this report we describe the implementation, simulation and analysis of an epidemics spreading over two networks taking into account mobility; in particular, we consider a disease spreading according to a stochastic SIRS process over two networks whose nodes travel with a certain probability according to a commuting pattern. The analysis of the results mainly focuses on how the probability of mobility and the differences in the structure of the networks affect the extinction and the recurrence of the disease.*

**Notebooks description:**
- SIRS_mobility_and_network_structure_EXPLAINED - first notebook produced, contains low-level details of how the simulation is done. All the functions have been migrated to SIRS.py module .
- SIRS_simulation - simulates a pair of scale-free and Erdosh-Renyi networks
- SIRS_simulation_SF - simulates a pair of scale-free networks with asymmetric initial conditions
- SIRS_simulation_SF_sym - simulates a pair of scale-free networks with symmetric initial conditions
- SIRS_analysis - contains all the analysis and visualization code

Additionaly part of the code has been copied to SIRS.py and SIRS_twoSF.py modules in order to have less packed notebooks.

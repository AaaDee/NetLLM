# NetLLM Reproduction Project

This is the repository for a research project on reproducing the key results of the SIGCOMM 2024 paper "[NetLLM: Adapting Large Language Models for Networking](https://dl.acm.org/doi/abs/10.1145/3651890.3672268)".

## Overview

This repository is a fork of the original [NetLLM Repositiory](https://github.com/duowuyms/NetLLM).

In addition to the original source code released by the NetLLM authors, this repository contains the following updates:

- Scripts for running the model testing and simulations with pre-determined settings
- Scripts for calculating the key results from the data
- Various dependency version and other minor updates to run the code
- An extension to run the benchmarking on LLama3 base model
  - Due to conflicting dependency versions, this extension can be found separately in the `llama3` branch.

Following the original NetLLM folder structure, the code for running and testing the three different tasks studied can be found in their respective folders. The only exception is the calculation of results, which is contained in the `result_calculations`

For additional documentation of the source code, please refer to the material provided in the original NetLLm repository

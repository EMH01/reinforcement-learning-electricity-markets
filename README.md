# Reinforcement Learning for Electricity Markets

Technical research on the design of reinforcement-learning systems for electricity-market trading, with emphasis on **environment design, point-in-time data, reward formulation, execution realism, risk, temporal backtesting, and the Iberian market context**.

> **Status:** Completed technical research review · Version 1.0 · October 2026. This is a public research report and has not been peer reviewed.

## Paper

**Aprendizaje por refuerzo para trading en mercados eléctricos: diseño del entorno, algoritmos, herramientas y evidencia aplicada**

[Read the manuscript](paper/reinforcement-learning-electricity-markets.md)

The review covers:

- MDP formulation for sequential electricity-market decisions
- observation, action, reward, and constraint design
- PPO, DDPG, TD3, SAC, and offline RL
- temporal architectures, multi-agent RL, imitation learning, and hierarchical RL
- Gymnasium, PettingZoo, Stable-Baselines3, RLlib, CleanRL, TorchRL, Tianshou, and related tooling
- market/asset/settlement simulation
- evidence from recent electricity-market applications
- point-in-time data and leakage prevention
- temporal backtesting, baselines, risk metrics, and reproducibility
- implications for the Iberian electricity market

## Sources and scope

The report is based on **scientific literature, official documentation, and public sources**.

## Why this matters

In reinforcement learning for electricity trading, algorithm choice is only one part of the problem. A policy can look successful while exploiting unrealistic execution, future information, weak baselines, or an incorrectly specified reward.

The central research question of this work is therefore:

> **What makes an RL environment for electricity-market decision-making methodologically defensible and operationally meaningful?**

## Scope and completion

This research review is **complete for its intended scope**: identifying and organizing the methodological, algorithmic, simulation, evaluation, and market-design considerations needed to reason about reinforcement learning for electricity-market trading.

It does **not** claim an original production trading strategy or an experimental benchmark.

Version 1.0 closes the research scope covered by this repository. Future changes, if any, are expected to be limited to corrections, reference maintenance, or publication metadata rather than an expansion of the investigation.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Author

**Esther M. Martin Hernández**

Data Scientist · AI Engineer · Computer Scientist
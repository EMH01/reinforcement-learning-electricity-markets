# Reinforcement Learning for Electricity Markets

Technical research on the design of reinforcement-learning systems for electricity-market trading, with emphasis on **environment design, point-in-time data, reward formulation, execution realism, risk, temporal backtesting, and the Iberian market context**.

> **Status:** Working paper / narrative technical review. This repository is a public research artifact and has not been peer reviewed.

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

## Public-scope statement

This public version is based exclusively on **scientific literature, official documentation, and open sources**.

It contains **no client or company data, proprietary code, business rules, private parameters, internal metrics, confidential architectures, or non-public results**. The research is presented as a general methodological study independent of any specific commercial implementation.

## Why this matters

In reinforcement learning for electricity trading, algorithm choice is only one part of the problem. A policy can look successful while exploiting unrealistic execution, future information, weak baselines, or an incorrectly specified reward.

The central research question of this work is therefore:

> **What makes an RL environment for electricity-market decision-making methodologically defensible and operationally meaningful?**

## Current scope

This repository currently contains the research review itself. It does **not** claim an original production trading strategy or client-derived empirical result.

A future public experimental companion may add:

- a Gymnasium electricity-market environment built from public data
- heuristic and no-trade baselines
- PPO / SAC / TD3 comparisons
- temporal train/validation/test splits
- walk-forward evaluation
- multiple random seeds
- imbalance and risk-aware rewards
- reproducible experiment tracking

## Citation

See [`CITATION.cff`](CITATION.cff).

## Author

**Esther M. Martin Hernández**

Data Scientist · AI Engineer · Computer Scientist
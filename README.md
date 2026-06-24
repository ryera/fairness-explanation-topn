# Behind the Re-Rank: Explaining Fairness in Top-N Recommendations

[![Conference](https://img.shields.io/badge/UMAP-2026-blue.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

This is the official repository containing the code, data, and supplementary material for the paper **"Behind the Re-Rank: Explaining Fairness in Top-N Recommendations"**, published in the *Proceedings of the 34th ACM Conference on User Modeling, Adaptation and Personalization (UMAP 2026)*.

**Publication Link:** https://dl.acm.org/doi/full/10.1145/3774935.3806165

## 📖 Abstract

Ensuring Provider Fairness in Recommender Systems often requires post-processing re-ranking, a mechanism that deliberately perturbs recommendation lists to satisfy exposure constraints. This intervention creates an interpretability gap, as users experience a loss in perceived utility without understanding the underlying ethical rationale. 

In this work, we propose a comprehensive framework to extract and evaluate semantic explanations for fairness-driven re-ranking:

* **Technical Audit:** First, we conduct a comparative technical audit across three datasets, three fairness metrics, and two architectures (Matrix Factorization vs. Variational Autoencoders). This reveals that deep generative models enable precise, attribute-level substitutions, whereas linear models rely on broader categorical displacements.
* **User Evaluation:** Second, we validate these insights through a pre-registered user study ($N=54$) comparing distinct explanation strategies.

**Key Findings:** Our results demonstrate that highly faithful explanations—those explicitly revealing the items demoted—trigger loss aversion and reduce satisfaction. Conversely, strategies that frame fairness as a positive side-effect of diversity significantly mitigate this negative impact. 

These findings suggest that effective fairness interfaces must decouple algorithmic fidelity from user-facing communication, prioritizing semantic framing that aligns ethical constraints with personal discovery.

## 📂 Repository Structure

Below is the description of the files and directories included in this repository, maintaining the structure from our original OSF submission:

* **`📦 data.zip`**: Raw datasets used for the experiments.
* **`💻 expl_fairness_source.py`**: Core Python script containing the complete implementation of the proposed framework (recommendation, fairness re-ranking, and rule extraction).
* **`🗂️ Detailed Results/`**
  * `offline_results_including_rules.zip`: Text outputs for each offline experimental setting, including the extracted decision rules.
  * `offline_tabular_results.zip`: Tabulated performance and fairness metrics for all settings.
  * `working_scenario.pdf`: Step-by-step working scenarios illustrating how the framework operates (Toy Example).
* **`👥 User Study Materials/`**
  * `user_study_scenarios.pdf`: The exact survey questions and scenarios presented to participants (administered via Qualtrics).
  * `user_study_statist_analysis.txt`: Full output log of the Linear Mixed Models (LMM) statistical tests.

> **Note on Requirements:** To run `expl_fairness_source.py`, ensure you have Python 3.x installed along with `cornac`, `pandas`, `numpy`, and `PuLP`.

## 🚀 Quick Start

### Requirements
The framework requires **Python 3.x**. Since the dependencies are minimal, you can install all of them directly via pip with a single command:

```bash
pip install cornac pandas numpy pulp
```

## 📝 Citation
If you use this code in your research, please cite the original publication:

@inproceedings{Yera2026behind,
  title={Behind the Re-Rank: Explaining Fairness in Top-N Recommendations},
  author={Yera, R. and Mart{\'\i}nez, L.},
  booktitle={Proceedings of the 34th ACM Conference on User Modeling, Adaptation and Personalization},
  pages={21--29},
  year={2026}
}

## 🎓 Funding Acknowledgment
Raciel Yera is supported by the European Union's Horizon Europe research and innovation program under the Marie Skłodowska-Curie Grant Agreement No. 101106164.

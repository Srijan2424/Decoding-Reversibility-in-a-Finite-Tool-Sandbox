# Project status, 15 November 2026

Srijan Chopra, Advanced Deep Neural Network.

- **RQ1:** Layer-wise probes, controls and reduced ablations on 1,200 sandbox examples. Three original scenario-grouped splits are retained. The extended controls were reduced after the main test had been run.
- **RQ2:** Qwen and Llama each answered 1,200 same-example sandbox queries. Locally rescored answers matched the Colab exports. Three-class stated accuracy averaged 23.8% and 44.4%, respectively. There were 513 invalid Qwen responses and zero invalid Llama responses.
- **RQ3:** Paired tool behaviour and stated knowledge on a separate 60-pair dataset. The presentation demo replays actual archived Qwen actions.
- **ScienceWorld:** The user approved a reduced protocol. Across 12 simulator runs there were three distinct kitchen state–action pairs, with one partial and two irreversible labels. The comparison is limited to the declared commands and physical fields.
- **Annotation:** The 12-case follow-up covers both environments. Agreement counts are descriptive. The ScienceWorld outcomes had been discussed before the human exercise. Qwen is also one of the evaluated models.
- **Transfer:** Feasibility audit completed. Both transfer directions remain unrun. One ScienceWorld scenario family and zero exact-recovery cases prevent a held-out evaluation. Dataset expansion was declined.
- **Original scope:** The full ScienceWorld benchmark, 2,000-example combined target, bidirectional transfer and a full independent combined annotation audit remain unmet.

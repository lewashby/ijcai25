# Integrating Answer Set Programming and Large Language Models for Enhanced Structured Representation of Complex Knowledge in Natural Language

[![DOI: 10.24963/ijcai.2025/482](https://img.shields.io/badge/DOI-10.24963/ijcai.2025/482-blue)](https://doi.org/10.24963/ijcai.2025/482)

Answer Set Programming (ASP) and Large Language Models (LLMs) have emerged as powerful tools in Artificial Intelligence, each offering unique capabilities in knowledge representation and natural language understanding, respectively. In this paper, we combine the strengths of the two paradigms with the aim of improving the structured representation of complex knowledge encoded in natural language. In a nutshell, the structured representation is obtained by combining syntactic structures extracted by LLMs and semantic aspects encoded in the knowledge base. The interaction between ASP and LLMs is driven by a YAML file specifying prompt templates and domain-specific background knowledge. The proposed approach is evaluated using a set of benchmarks based on a new dataset obtained from problems of ASP Competitions. The results of our experiment show that ASP can sensibly improve the F1-score, especially when relatively small models are used.

## Run Experiments

Install [poetry](https://python-poetry.org/docs/#installation).

Inside the project folder run:

```bash
poetry shell
```

Then:

```bash
poetry install
```

There is CLI with for running the experiments for llmasp:

```bash
run-experiment --help
```

For further help with a single command use:

```bash
run-experiment <required_options> <command> --help
```

For running a single experiment with LLMASP use:

```bash
run-experiment <options> llmasp-run-selected <arguments>
```

For running all the experiments with LLMASP use:

```bash
run-experiment <options> llmasp-full-test <arguments>
```

Example using one of the behaviors files in the specifications folder:

```bash
run-experiment -m llama3.1:70b -bf specifications/behaviors/behavior_second_report_v6 llmasp-full-test
```

Example for a specific behavior file and domain:

```bash
run-experiment -m llama3.1:70b -bf specifications/behaviors/behavior_second_report_v6 llmasp-run-selected -q 1
```

The commands for running the experiments with Llama models are similar to those above.

## Authors

* Mario Alviano
* Lorenzo Grillo
* Fabrizio Lo Scudo
* Luis Angel Rodriguez Reiners

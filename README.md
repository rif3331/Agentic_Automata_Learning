# Agentic Automata Learning

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)]()
[![Website](https://img.shields.io/badge/website-blue.svg)](https://agentic-automata-learning.onrender.com)

Agentic Automata Learning is an evaluation framework for studying Large Language Model (LLM) agents. The framework investigates whether agents can infer a hidden structure of an environment through interaction, information gathering, and iterative hypothesis refinement.

## Components

- 📄 **Research Paper** - Introduces the Agentic Automata Learning framework and presents experimental results on modern LLM agents.
- 🌐 **Web Interface** - Agentic Automata Learning Runner, an interactive interface for launching experiments, monitoring agent interactions, and visualizing learning trajectories. Users can explore and run experiments directly in the browser for free, with no installation or API key required.
- 💻 **Source Code** - Complete implementation of the evaluation framework, experiment runner, task generation tools, and analysis utilities.

## Web Interface

The Agentic Automata Learning Runner provides an interactive web interface for configuring, running, and analyzing Agentic Automata Learning experiments directly from the browser.

<p align="center">
  <img src="docs/images/web_interface_main.png" width="450">
</p>

The interface allows users to select the API provider and model used during the experiment. By default, the runner is configured to use **Gemini 3.1 Flash Lite**, which is available free of charge through a shared daily budget of **$40** across all users of the demo. For other models, users are required to provide their own API key.

Users can choose between two sources for the hidden target automaton:

- **Dataset DFA** – sample a target DFA from the same automatically generated benchmark distribution used in our experiments.
- **User Regular Expression → DFA** – define a custom target automaton by providing a regular expression, which is automatically converted into a minimal DFA.

Once configured, experiments can be launched directly from the interface while monitoring the interaction between the agent and the oracle in real time.

### Advanced Experiment Options

Some experiment parameters are hidden under the *Experiment Options* section because the default values correspond to the configuration used throughout the paper's evaluation.

- **Alphabet Size** – controls the size of the DFA alphabet used during generation. Larger alphabets generally increase the complexity of the learning task.

- **Counterexample Mode** – determines how counterexamples are selected when an equivalence query fails. The default setting returns short counterexamples, matching the protocol used in our experiments.

- **Algorithm Approximation Ratio** – controls the query budget allocated to the agent. The budget is defined relative to the number of queries required by classical active automata learning algorithms (L* and TTT). The default value of **2** corresponds to the experimental setup described in the paper, where agents receive up to twice the query budget required by the stronger classical baseline.

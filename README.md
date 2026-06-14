# Agentic Automata Learning

[![Website](https://img.shields.io/badge/website-blue.svg)]([https://agentic-automata-learning.onrender.com](https://reefmenaged.github.io/Agentic_Automata_Learning))

Agentic Automata Learning is an evaluation framework for studying Large Language Model (LLM) agents. The framework investigates whether agents can infer a hidden structure of an environment through interaction, information gathering, and iterative hypothesis refinement.


## Source Code

### Requirements

- Python 3.9 or higher (recommended: Python 3.11)
- pip
- Git

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/reefmenaged/Agentic_Automata_Learning.git
```

Then enter the project directory:

```bash
cd Agentic_Automata_Learning
```

#### 2. Create a Virtual Environment

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Supported API Providers

The experiment runner supports multiple API providers. The provider is selected using `--api-provider`, while the model is selected using `--model-name`.

Supported providers:

- `google`
- `openai`
- `deepseek`
- `anthropic`
- `together`

### Choosing a Target Automaton

Each experiment requires selecting a hidden target automaton. The framework supports two target sources.

#### 1. Dataset DFA

This option samples a target DFA from the same automatically generated distribution used throughout the experiments in the paper.

Required arguments:

- `--target-source dataset`
- `--n-states` — number of states in the hidden minimal DFA
- `--seed` — random seed used to select or generate the target DFA

Example:

```bash
python main.py ^
  --api-provider google ^
  --model-name gemini-3.1-flash-lite-preview ^
  --api-key YOUR_API_KEY ^
  --target-source dataset ^
  --n-states 4 ^
  --seed 42
```

#### 2. Regular Expression Target

This option allows users to define a custom target automaton by providing a regular expression. The expression is automatically converted into a minimal DFA and used as the hidden target.

Required arguments:

- `--target-source regex`
- `--regex` — regular expression defining the target language

Example:

```bash
python main.py ^
  --api-provider google ^
  --model-name gemini-3.1-flash-lite-preview ^
  --api-key YOUR_API_KEY ^
  --target-source regex ^
  --regex "(a|b)*abb"
```

### Optional Arguments

The following arguments can be added to either target source.

#### Alphabet Size

```bash
--alphabet-size 2
```

Controls the size of the DFA alphabet. This parameter is only relevant when using a Dataset DFA.

#### Counterexample Mode

```bash
--counterexample-mode "deterministic short counterexample"
```

Controls how counterexamples are selected when an equivalence query fails.

Available modes:

- `deterministic short counterexample (defult)` — selects a short counterexample deterministically from a small candidate set.
- `minimal counterexample` — always returns the shortest possible counterexample.

#### Algorithm Approximation Ratio

```bash
--algorithm-approximation-ratio 2
```

Controls the query budget allocated to the LLM agent. The budget is computed relative to the number of queries required by the stronger classical baseline between L* and TTT.

#### Output Directory

```bash
--output-dir runs
```

Directory where all experiment outputs are saved.

#### Results CSV

```bash
--experiment-csv results.csv
```

Name of the CSV file used to store experiment results.

### Supported Models

#### Google

```text
gemini-3.1-flash-lite-preview
gemini-3.1-pro-preview
gemini-3-flash-preview
```

#### OpenAI

```text
gpt-5.4
```

#### DeepSeek

```text
deepseek-v4-pro
```

#### Anthropic

```text
claude-sonnet-4-6
```

#### Together

```text
meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### Model Configuration

Some models support additional provider-specific configuration parameters beyond the model name itself. These parameters can be specified inside parentheses following the model name.

For example:

```text
deepseek-v4-pro(extra_body.thinking.type=enabled, reasoning_effort=high)

gemini-3-flash-preview(thinking_level=high)
```

The framework interprets the text before the parentheses as the model name and the contents of the parentheses as additional configuration parameters.

Any configuration parameter specified inside the parentheses is forwarded directly to the provider's API request. This allows users to take advantage of any advanced options supported by the provider, such as reasoning budgets, thinking modes, or other model-specific capabilities, without modifying the framework's source code.

### Output

All experiment artifacts are saved inside the selected output directory. By default:

```text
runs/
```

The output directory contains both per-game analyses and aggregate statistics across all experiments.

#### Results Table (`results.csv`)

The central output file is `results.csv`, which contains one row per experiment.

Important columns include:

| Column | Description |
|----------|-------------|
| `llm_model` | Evaluated language model. |
| `number_of_states` | Number of states in the hidden minimal DFA. |
| `alphabet_size` | Alphabet size used during generation. |
| `seed` | Random seed used to generate or select the target automaton. |
| `counterexample_mode` | Counterexample selection strategy. |
| `max_tool_calls` | Query budget allocated to the model. |
| `llm_total_queries` | Number of queries performed by the model. |
| `total_game_time_s` | Total runtime of the experiment. |
| `game_token_tuple` | Token usage statistics. |
| `conversation_link` | Link to the full HTML report. |
| `llm_gold_step` | Step at which the target DFA was identified. |
| `llm_reached_gold_triangle` | Whether the target DFA was successfully recovered. |
| `llm_inefficient_steps` | Number of non-informative queries. |
| `llm_symdiff_similarity_by_step` | Similarity trajectory of the model's hypotheses. |
| `llm_hypothesis_monotonicity_broken` | Whether the hypothesis sequence violated monotonicity. |
| `llm_eq_count_gt_target_states` | Whether the number of equivalence queries exceeded the number of target states. |

#### HTML Reports

The `html/` directory contains a complete report for every game.

Each report includes:

- Full interaction history between the agent and the oracle.
- Membership Queries (MQs) and Equivalence Queries (EQs).
- Visualizations of all proposed DFAs.
- Similarity analyses between hypotheses and the target DFA.
- Non-informative query analyses.
- Passive learning analyses using RPNI, EDSM, and Blue-Fringe.
- Comparisons with L* and TTT.

These reports provide a complete record of the agent's behavior and learning process.

#### Additional Output Directories

- `DFA/` — visualizations of target and hypothesis DFAs.
- `evaluations/` — evaluation artifacts generated during the runs.
- `L_star_comparisons/` — comparisons with the L* algorithm.
- `TTT_comparisons/` — comparisons with the TTT algorithm.

### Generating the Paper Figures

After collecting experiment results, aggregate analyses and visualizations can be generated directly from the results table.

Run:

```bash
python app/create_graphs.py runs/results.csv
```

Replace `runs/results.csv` with the path to the results table you would like to analyze.

The script generates:

- PDF reports containing the figures presented in the paper.
- Success-rate analyses.
- Query-efficiency comparisons against L* and TTT.
- Hypothesis-similarity analyses.
- Runtime statistics.
- Token-usage analyses.
- Cost analyses.
- Additional visualizations computed from the collected experiment results.
- Cost analyses.
- Additional visualizations computed from the collected experiment results.

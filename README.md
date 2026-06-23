# World Cup 2026 Prediction System 🏆

A comprehensive prediction pipeline and Monte Carlo simulator for the FIFA World Cup 2026. This project leverages historical international football data, advanced statistical models (Dixon-Coles), machine learning (XGBoost), and an optional LLM Agent to forecast match outcomes and tournament champions.

## Features

1. **Data Pipeline**: Automatic downloading and loading of historical match data (including or excluding friendlies).
2. **Elo Ratings**: Computes historical and current Elo ratings for national teams, including home advantage adjustments.
3. **Feature Engineering**: Calculates form windows and advanced team statistics.
4. **Multiple Predictive Models**:
   - **XGBoost**: Tree-based model with automated hyperparameter optimization via Optuna.
   - **Dixon-Coles (Poisson)**: Statistical model specifically designed for football score prediction, incorporating time-decay (`xi`).
   - **LLM Agent (Optional)**: Employs LangChain and LangGraph to synthesize predictions when base models disagree significantly (confidence threshold). Supports OpenAI and NVIDIA NIM.
5. **Rigorous Backtesting**: Evaluates model performance using historical World Cup data (e.g., training on <2018 or <2022 and testing on subsequent tournaments). Generates comparison plots.
6. **Monte Carlo Simulation**: Runs thousands of simulations for the World Cup 2026 bracket.
7. **Live Data Integration**: Optionally fetches live WC 2026 results from ESPN to update probabilities mid-tournament.

## Requirements

- Python >= 3.14
- Dependencies managed via `requirements.txt` or `pyproject.toml` (managed by `uv`).
- Nvidia NIM API key for LLM Agent predictions (if using NVIDIA NIM). obtainable from [NVIDIA Build](https://build.nvidia.com/).

Key packages:
- `pandas`, `numpy`, `scipy`
- `xgboost`, `scikit-learn`, `optuna`
- `matplotlib`, `seaborn`
- `langchain`, `langgraph` (for the Agent model), `langchain-nvidia-ai-endpoints`(if using NVIDIA NIM)

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd WorldCup-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR using uv
   uv sync
   ```

## Configuration

The pipeline is heavily configurable via `config.yaml`. You can adjust:
- Data boundaries (`start_year`, `include_friendlies`)
- Elo parameters (`k_base`, `home_advantage`)
- Model parameters (`n_optuna_trials`, `xi` for Dixon-Coles)
- Agent settings (LLM model, API keys, disagreement threshold)
- Evaluation cutoffs and Simulation counts

## Usage

### Full Pipeline
Run the complete pipeline: data loading -> elo -> feature engineering -> training -> backtesting -> simulation.
```bash
python main.py
```

### Skip Evaluation
If you want to skip historical backtesting and go straight to training on all data and simulating WC 2026:
```bash
python main.py --skip-backtest
```

### Single Match Prediction
Quickly predict the outcome between two specific teams:
```bash
python main.py --match "Brazil" "France"
```

## Output

All results are saved in the `output/` directory:
- `comparison.csv` and comparison plots for model evaluation.
- `sim_XGBoost.csv`, `sim_Dixon-Coles.csv` containing the champion, finalist, and semi-finalist probabilities for each team across simulations.

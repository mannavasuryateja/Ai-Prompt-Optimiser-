# AI Prompt Optimizer

This project trains a model to shorten prompts while improving or maintaining the quality of LLM outputs.

## Features
- T5-based prompt compression
- LLM evaluation via OpenRouter.ai (free tier)
- Gradio demo interface
- Reward model with semantic similarity + token penalty

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get a free API key from https://openrouter.ai and set it in `utils.py`.

3. Train the optimizer:
```bash
python train.py
```

4. Launch the demo:
```bash
python app.py
```

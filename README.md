# Breaking AI Systems: From Image Classifiers to LLM Agents

**Guest Lecture — [CS 6501: Security of AI Systems](https://dartlab.org/cs6501-sp26/schedule/#:~:text=Breaking%20AI%20Systems%3A%20From%20Image%20Classifiers%20to%20LLM%20Agents) | University of Virginia, February 2026**

**Raja Sekhar Rao Dheekonda** | Distinguished Engineer @ [Dreadnode](https://dreadnode.io)

---

Hands-on demos and a case study covering adversarial attacks across the full AI stack — from adversarial perturbations on image classifiers to jailbreaking LLMs and multimodal systems.

## Demos

| # | Notebook | Description |
|---|----------|-------------|
| Setup | [00_setup_verification.ipynb](demos/00_setup_verification.ipynb) | Verify your environment, dependencies, and API keys are configured correctly before running demos |
| 1 | [adversarial_model_evasion.ipynb](demos/adversarial_model_evasion.ipynb) | Force a MobileNetV2 classifier to misclassify a wolf as a "Granny Smith apple" using white-box PGD and black-box HopSkipJump attacks |
| 2 | [llm_text_model_probing.ipynb](demos/llm_text_model_probing.ipynb) | Use Tree of Attacks with Pruning (TAP) to automatically discover jailbreak prompts that bypass an LLM's safety guardrails |
| 3 | [multimodal_probing.ipynb](demos/multimodal_probing.ipynb) | Exploit cross-modal attack surfaces by splitting harmful intent across text and image inputs to bypass multimodal safety filters |
| Case Study | [case_study_186_jailbreaks.ipynb](demos/case_study_186_jailbreaks.ipynb) | Analyze results from running TAP, GOAT, and Crescendo at scale — 186 jailbreaks discovered in 137 minutes of automated red teaming |

## Prerequisites

### Accounts

1. **Dreadnode Platform** — [platform.dreadnode.io](https://platform.dreadnode.io)
   - Sign up for a free account
   - Copy your API key from the dashboard
   - Required for Demo 1 (Crucible challenge) and result tracking

2. **Groq** — [console.groq.com](https://console.groq.com)
   - API key for Llama 4 Maverick model access
   - Used for all LLM demos (attacker, evaluator, and target)

### System Requirements

- Python 3.10 — 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

### 1. Install uv (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository

```bash
git clone https://github.com/rdheekonda/breaking-ai-systems.git
cd breaking-ai-systems
```

### 3. Create the environment and install dependencies

```bash
uv sync
```

### 4. Set your API keys

Copy the example environment file and fill in your keys:

```bash
cp .env-example .env
# Edit .env with your actual API keys
```

### 5. Launch Jupyter

```bash
uv run jupyter notebook demos/
```

### 6. Verify setup

Open and run `demos/00_setup_verification.ipynb` to confirm everything is working.

## Project Structure

```
breaking-ai-systems/
├── core/
│   ├── __init__.py         # Public API — all exports
│   ├── models.py           # Shared model loading + preprocessing (MobileNetV2)
│   ├── pgd.py              # White-box PGD attack (PGDResult + run_pgd)
│   ├── hop_skip_jump.py    # Black-box HopSkipJump attack (HSJResult + run_hsj)
│   ├── display.py          # Shared Rich comparison tables
│   ├── visual.py           # Matplotlib visualization helpers
│   ├── utils.py            # Crucible flag submission helpers
│   ├── tap.py              # Tree of Attacks with Pruning (TAP) implementation
│   └── transforms.py       # Image transforms for multimodal attacks
├── data/
│   ├── reference.png       # Wolf image for Demo 1 (adversarial evasion)
│   └── meth.png            # Reference image for Demo 3 (multimodal probing)
├── demos/
│   ├── 00_setup_verification.ipynb
│   ├── adversarial_model_evasion.ipynb
│   ├── llm_text_model_probing.ipynb
│   ├── multimodal_probing.ipynb
│   └── case_study_186_jailbreaks.ipynb
├── .env-example            # Template for API keys
└── pyproject.toml
```

## Models Used

All LLM calls are routed through [LiteLLM](https://docs.litellm.ai/), providing a unified interface to 100+ providers. Swap the `groq/` prefix (e.g., `openai/`, `anthropic/`, `azure/`, `bedrock/`) to target any model.

| Model | Provider | Role |
|-------|----------|------|
| Llama 4 Maverick 17B 128E Instruct | Groq | Attacker, Evaluator, Target (Demos 2–3, Case Study) |
| MobileNetV2 (ImageNet) | Crucible / Dreadnode | Target (Demo 1) |

## Frameworks Referenced

- **[MITRE ATLAS](https://atlas.mitre.org/)** — Adversarial Threat Landscape for AI Systems
- **[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)** — Security risks specific to LLM deployments

## Resources

- **Blog**: [186 Jailbreaks: Applying MLOps to AI Red Teaming](https://dreadnode.io/blog/186-jailbreaks-applying-mlops-to-ai-red-teaming)
- **Crucible CTF**: [platform.dreadnode.io/crucible](https://platform.dreadnode.io/crucible)

## License

MIT

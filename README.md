[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dreadnode](https://img.shields.io/badge/Dreadnode-Platform-black.svg)](https://dreadnode.io)

# Breaking AI Systems: From Image Classifiers to LLM Agents

Hands-on adversarial attacks across the full AI stack -- from white-box image perturbations to LLM jailbreaking and multimodal exploitation -- plus a 186-jailbreak case study.

> **Guest Lecture** -- [CS 6501: Security of AI Systems](https://dartlab.org/cs6501-sp26/schedule/#:~:text=Breaking%20AI%20Systems%3A%20From%20Image%20Classifiers%20to%20LLM%20Agents) | University of Virginia, February 2026
>
> **Raja Sekhar Rao Dheekonda** | Distinguished Engineer @ [Dreadnode](https://dreadnode.io)
>
> [**Slides (PDF)**](Breaking%20AI%20Systems%20-%20UVA%20Guest%20Lecture%202026%20.pdf)

---

## Prerequisites

### Accounts

1. **Dreadnode Platform** -- [platform.dreadnode.io](https://platform.dreadnode.io)
   - Sign up for a free account
   - Copy your API key from the dashboard
   - Required for Demo 1 (Crucible challenge) and result tracking

2. **Groq** -- [console.groq.com](https://console.groq.com)
   - API key for Llama 4 Maverick model access
   - Used for all LLM demos (attacker, evaluator, and target)

### System Requirements

- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

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

---

## Demos

| # | Notebook | What It Does |
|---|----------|--------------|
| Setup | [00_setup_verification.ipynb](demos/00_setup_verification.ipynb) | Verify your environment, dependencies, and API keys are configured correctly before running demos |
| 1 | [adversarial_model_evasion.ipynb](demos/adversarial_model_evasion.ipynb) | Force a MobileNetV2 classifier to misclassify a wolf as a "Granny Smith apple" using white-box PGD and black-box HopSkipJump attacks |
| 2 | [llm_text_model_probing.ipynb](demos/llm_text_model_probing.ipynb) | Use Tree of Attacks with Pruning (TAP) to automatically discover jailbreak prompts that bypass an LLM's safety guardrails |
| 3 | [multimodal_probing.ipynb](demos/multimodal_probing.ipynb) | Exploit cross-modal attack surfaces by splitting harmful intent across text and image inputs to bypass multimodal safety filters |
| Case Study | [case_study_186_jailbreaks.ipynb](demos/case_study_186_jailbreaks.ipynb) | Analyze results from running TAP, GOAT, and Crescendo at scale -- 186 jailbreaks discovered in 137 minutes of automated red teaming |

---

## Attack Surface Coverage

```text
Image Classifiers --> LLM Text --> Multimodal
```

| Demo | Attack Type | Technique | Access Level |
|------|-------------|-----------|--------------|
| 1 | Adversarial perturbation | PGD | White-box (full gradients) |
| 1 | Adversarial evasion | HopSkipJump | Black-box (query only) |
| 2 | Jailbreak generation | TAP | API access |
| 3 | Cross-modal exploitation | Text + image splitting | API access |

---

## Models Used

All LLM calls are routed through [LiteLLM](https://docs.litellm.ai/), providing a unified interface to 100+ providers. Swap the `groq/` prefix (e.g., `openai/`, `anthropic/`, `azure/`, `bedrock/`) to target any model.

| Model | Provider | Role |
|-------|----------|------|
| Llama 4 Maverick 17B 128E Instruct | Groq | Attacker, Evaluator, Target (Demos 2--3, Case Study) |
| MobileNetV2 (ImageNet) | Crucible / Dreadnode | Target (Demo 1) |

---

## Core Dependencies

| Library | Purpose |
|---------|---------|
| [LiteLLM](https://docs.litellm.ai/) >= 1.40 | Unified LLM API interface |
| [torchvision](https://pytorch.org/vision/) >= 0.17 | Pre-trained models and image transforms |
| [Adversarial Robustness Toolbox](https://adversarial-robustness-toolbox.readthedocs.io/) >= 1.18 | PGD, HopSkipJump, and other attack implementations |
| [Rich](https://rich.readthedocs.io/) >= 13.0 | Terminal output formatting |

---

## Project Structure

```
breaking-ai-systems/
├── core/
│   ├── __init__.py         # Public API — all exports
│   ├── models.py           # Shared model loading + preprocessing (MobileNetV2)
│   ├── pgd.py              # White-box PGD attack (PGDResult + run_pgd)
│   ├── hop_skip_jump.py    # Black-box HopSkipJump attack (HSJResult + run_hsj)
│   ├── tap.py              # Tree of Attacks with Pruning (TAP) implementation
│   ├── multimodal.py       # Multimodal attack interface (cross-modal probing)
│   ├── display.py          # Shared Rich comparison tables
│   ├── visual.py           # Matplotlib visualization helpers
│   ├── transforms.py       # Image transforms for multimodal attacks
│   └── utils.py            # Crucible flag submission helpers
├── data/
│   ├── reference.png       # Wolf image for Demo 1 (adversarial evasion)
│   └── meth.png            # Reference image for Demo 3 (multimodal probing)
├── demos/
│   ├── 00_setup_verification.ipynb
│   ├── adversarial_model_evasion.ipynb
│   ├── llm_text_model_probing.ipynb
│   ├── multimodal_probing.ipynb
│   └── case_study_186_jailbreaks.ipynb
├── Breaking AI Systems - UVA Guest Lecture 2026 .pdf
├── .env-example            # Template for API keys
└── pyproject.toml
```

---

## Resources

- **Slides**: [Breaking AI Systems -- UVA Guest Lecture 2026 (PDF)](Breaking%20AI%20Systems%20-%20UVA%20Guest%20Lecture%202026%20.pdf)
- **Blog**: [186 Jailbreaks: Applying MLOps to AI Red Teaming](https://dreadnode.io/blog/186-jailbreaks-applying-mlops-to-ai-red-teaming)
- **Full eval notebook**: [AI Red Teaming Eval](https://github.com/dreadnode/sdk/blob/main/examples/airt/ai_red_teaming_eval.ipynb) -- complete implementation orchestrating TAP, GOAT, and Crescendo at scale
- **Crucible CTF**: [platform.dreadnode.io/crucible](https://platform.dreadnode.io/crucible)

---

## Frameworks Referenced

| Framework | Focus |
|-----------|-------|
| [MITRE ATLAS](https://atlas.mitre.org/) | Adversarial Threat Landscape for AI Systems |
| [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Security risks specific to LLM deployments |

---

## License

[MIT](LICENSE)

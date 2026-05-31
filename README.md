# 🤖 Model Evaluation & Lifecycle Platform

[![CI](https://github.com/kallurayaankit/model-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/kallurayaankit/model-hub/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-green?logo=huggingface)](https://kallurayaankit-model-hub.hf.space)

A web‑based “Model Hub” that lets teams register models, run automated evaluation suites, generate shareable reports, and make go/no‑go decisions — all from a single interface.

---

## 📌 Features

- **Model Registry UI** – browse registered models and their versions
- **Automated Evaluation Engine** – accuracy, ROC‑AUC, fairness (group parity), slice‑based metrics, and latency
- **Report Generator** – HTML report with plots (confusion matrix, ROC curve) and a go/no‑go recommendation
- **Webhook Simulation** – Slack‑style notification and auto‑PR trigger when a model passes
- **Streamlit Dashboard** – clean, interactive UI
- **CI/CD Ready** – GitHub Actions validates the pipeline on every push

---

---

## ⚡ Quick Start (Local)

### 1. Set up environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

---

## 🟢 2. Add a GitHub Actions workflow (makes the badge green)

Create the workflow file:

```cmd
mkdir .github\workflows
notepad .github\workflows\ci.yml
name: Validate Model Hub

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Check Python syntax
        run: |
          python -m py_compile app.py
          python -m py_compile evaluations/evaluator.py
          python -m py_compile evaluations/report.py
git add README.md .github\workflows\ci.yml
git commit -m "Add README with CI badge and GitHub Actions workflow"
git push
pip install -r requirements.txt
## 📁 Project Structure

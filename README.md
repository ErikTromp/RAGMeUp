# 🚀 RAGMeUp

> A simple and extensible framework to build RAG (Retrieval-Augmented Generation) applications fast.

[![License](https://img.shields.io/github/license/FutureClubNL/RAGMeUp?style=flat-square)](https://github.com/FutureClubNL/RAGMeUp/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/FutureClubNL/RAGMeUp?style=social)](https://github.com/FutureClubNL/RAGMeUp/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/FutureClubNL/RAGMeUp?style=flat-square)](https://github.com/FutureClubNL/RAGMeUp/issues)
[![Docs](https://img.shields.io/badge/docs-Docusaurus-blueviolet?logo=readthedocs&style=flat-square)](https://ragmeup.futureclub.nl)

---

## ⚡ TL;DR – Installation & Quickstart

**Server**
```
# Clone the repo
git clone https://github.com/FutureClubNL/RAGMeUp.git
cd RAGMeUp/server

# Set up a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the RAG server
python server.py
```

**UI**
```
# In RAGMeUp folder in a separate shell, make sure you have JDK17+
cd ui
sbt run
```

## 📘 Documentation
Full setup instructions, architecture docs, API references, and guides available at:
👉 https://ragmeup.futureclub.nl


## 🧠 Why RAGMeUp?

⚙️ Modular: Use your own chunkers, vectorstores, or retrievers

🚀 Fast to prototype: Focus on your RAG logic, not boilerplate

🧩 Flexible: Plug-and-play architecture

## 🤝 Contributing
We welcome pull requests, feedback, and ideas.
Open an issue or start a discussion to get involved.
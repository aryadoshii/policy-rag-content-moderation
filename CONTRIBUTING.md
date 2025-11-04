# Contributing to Policy-Grounded RAG Content Moderation

Thank you for your interest in contributing! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

We welcome feature requests! Please open an issue describing:
- The proposed feature
- Use case and benefits
- Possible implementation approach

### Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a PR with clear description

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions focused and concise

### Testing
```bash
pytest tests/
```

## Questions?

Contact the maintainers at arya.doshi22@vit.edu
```

---

### **5. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Models (too large for git)
models/*.pt
models/*.bin
models/rag_embeddings/

# Data (add to .gitattributes for LFS if needed)
data/*.csv
!data/sample.csv

# Results
results/*.png
results/*.pdf

# OS
.DS_Store
Thumbs.db

# Logs
*.log

---
name: Issue Report
about: Report a issue or unexpected behavior
title: "[Issue] "
labels: issue
assignees: ''
---
## 🐛 Issue Description

A clear and concise description of what the issue is.

---
## ✅ Steps to Reproduce

Steps to reproduce the behavior:
1. Command '...'
2. Command '....'
3. Scroll down to log '....'
4. See error

---
## 🤔 Expected Behavior

What you expected to happen.

---
## 📸 Screenshots

If applicable, add screenshots to help explain your problem.

---
## 🧪 Environment

Please provide the following details:

- System Info
  - OS: Ubuntu 22.04.5 LTS
  - GPU: RTX 4090D (24GB RAM version)
  - CUDA: 12.8
  - GPU Driver: 570.124.04

- nvidia-smi, eg.

```bash
Fri Apr 25 06:22:23 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.04             Driver Version: 570.124.04     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:03:00.0 Off |                  N/A |
| 32%   44C    P0            116W /  350W |   10950MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          673227      C   python                                10536MiB |
|    0   N/A  N/A          870880    C+G   ./kit/kit                               258MiB |
+-----------------------------------------------------------------------------------------+
```

---

## 📎 Additional Context
Add any other context about the problem here.

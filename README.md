# Codebase - Kernel-Based Sensitivity Analysis
Codebase accompanying the master’s thesis "A Kernel Framework for Sensitivity Analysis of Set-Valued Processes: Theoretical Foundations and Numerical Validation" by Farbod Chamanian (M.Sc. Mathematics in Science and Engineering), providing implementations and experiments for reproducing all results.

The setup uses **Miniconda**, the 2019 legacy **FEniCS Project**, and a set of pinned Python dependencies. 

---
## Prerequisites

### 1. Install Miniconda
If you don’t already have Conda installed, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

**Linux / macOS**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

**Windows**
Download the Miniconda installer for Windows here: https://docs.conda.io/en/latest/miniconda.html#windows-installers

After installation, make sure conda is available by running the followig bash command:

conda --version

This repo includes an environment.yml file specifying all dependencies.

### 2. Create the environment
This repo includes an environment.yml file specifying all dependencies.
To create the environment run the following bash command:

conda env create -f environment.yml
conda activate thesis-repro

Register the kernel with Jupyter by running the following bash command:

python -m ipykernel install --user --name thesis-repro

### 3. Install FEniCS
The legacy 2019 version of FEniCS can be installed via Conda Forge by using the following bash command, after activating the conda environment:

conda install -c conda-forge fenics


### 4. Running Juypter notebooks
Either run the notebooks in the supporting IDE (e.g. VSCode) or launch Jupyter Lab inside the environment by running the following bash command and selecting the thesis-repo kernel when running the notebooks:

jupyter lab



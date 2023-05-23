# ARGS Node Classification (GCN) on Cora, Citeseer Datasets under PGD attack

This repository contains the implementation of ARGS Node Classification using GCN on Cora, Citeseer datasets under PGD attack.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Baseline Results](#baseline-results)

<a name="installation"></a>
## Installation

This project requires Python 3.6 and the following Python libraries installed:

- torch (>=1.4.0)
- dgl-cu101 (0.4.2)
- scikit-learn
- networkx
- scipy
- matplotlib
- pickle
- argparse
- numpy
- warnings

To install these dependencies, navigate to your project's directory and run the following command:

```bash
pip install -r requirements.txt
```
<a name="usage"></a>
## Usage

<<<<<<< HEAD
You can run the main script with various settings. Below is an example:

### Running GCN with PGD Attack on Cora and Citeseer Datasets
=======
## Evaluation 
The main script can be run with different settings. Here is the usage:

### GCN PGD cora, citeseer
>>>>>>> origin/master
```
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.05
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.1
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.15
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.2

python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.05
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.1
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.15
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.2
```
<a name="baseline-results"></a>
## Baseline Results
<<<<<<< HEAD

Here are the baseline accuracies for GCN on the Cora and Citeseer datasets under PGD attack for comparison:

### GCN with PGD Attack on Cora Dataset

| Perturbation Rate | Accuracy |
|-------------------|----------|
| 0.05              | 78.81    |
| 0.1               | 78.46    |
| 0.15              | 77.93    |
| 0.2               | 77.99    |

### GCN with PGD Attack on Citeseer Dataset

| Perturbation Rate | Accuracy |
|-------------------|----------|
| 0.05              | 73.28    |
| 0.1               | 73.96    |
| 0.15              | 74.62    |
| 0.2               | 73.52    |
=======
The following are the baseline accuracies of GCN on the Cora and Citeseer datasets under PGD attack for comparison:

### GCN PGD Cora
```
ptb_rate=0.05: 78.81
ptb_rate=0.1: 78.46
ptb_rate=0.15: 77.93
ptb_rate=0.2: 77.99
```
### GCN PGD Citeseer
```
ptb_rate=0.05: 73.28 
ptb_rate=0.1: 73.96
ptb_rate=0.15: 74.62
ptb_rate=0.2: 73.52
```
>>>>>>> origin/master

# ARGS Node Classification (GCN) on Cora, Citeseer Datasets under PGD attack

## Requirements
The following packages are needed to run the scripts:

- `python==3.6`
- `torch >= 1.4.0`
- `dgl-cu101==0.4.2`
- `scikit-learn`
- `networkx`
- `scipy`
- `matplotlib`
- `pickle`
- `argparse`
- `numpy`
- `warnings`

Install these packages using pip:

```bash
pip install -r requirements.txt
```

## Evaluation 
The main script can be run with different settings. Here is the usage:

### GCN PGD cora, citeseer
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

## Baseline Results
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

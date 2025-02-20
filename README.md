# TGOpt: Redundancy-Aware Optimizations for Temporal Graph Attention Networks

This repository is the artifact for our TGOpt paper (PPoPP 2023). For the
archived artifact, see our Zenodo record.

[[Paper][pdf], [Zenodo][zen]]

## Requirements

* Python 3.7, PyTorch 1.12, dependencies in `requirements.txt`
* g++ >= 7.2, OpenMP >= 201511, Intel TBB >= 2020.1 (for the C++ extension)

This repo provides scripts for setting up docker (see `Dockerfile`) or an AWS
GPU instance (see `scripts/setup-aws-ec2.sh`). Otherwise, we recommend using
conda to create a virtual Python environment. If needed, manually compile the
extension after installing dependencies:

```
$ pip install -r requirements.txt
$ cd extension && make && cd ..
```

## Datasets

The datasets we used in our experiments are provided in our [Zenodo][zen]
artifact under the `data/` subdirectory. You can prepare the datasets yourself
using the `data-*` scripts:

```
$ ./data-download.sh <name>
$ python data-reformat.py -d <name>
$ python data-process.py -d <name>
```

## Trained models

The models used in our experiments are provided in our [Zenodo][zen] artifact
under the `saved_models/` subdirectory. You can train different models using
the `train.py` script (be sure to use the same `--model` flag in later
inference commands):

```
$ python train.py -d <name> --model <prefix> ... [--gpu 0]
```

## Running experiments

The `inference.py` script is the main entry-point for our experiments. To kick
the tires, run a small experiment:

```
$ python inference.py -d snap-msg --model tgat --prefix test --opt-all [--gpu 0]
```

Use the `-h` flag to see all available options. To reproduce our experiments,
see Appendix A in our [paper][pdf].

## Citation

You can cite our paper with the following:

```bibtex
@inproceedings{tgopt:ppopp23,
  author = {Wang, Yufeng and Mendis, Charith},
  title = {{TGOpt}: Redundancy-Aware Optimizations for Temporal Graph Attention Networks},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages = {354–368},
  numpages = {15},
  series = {PPoPP '23}
}
```

[zen]: https://zenodo.org/record/7328505
[pdf]: https://dl.acm.org/doi/abs/10.1145/3572848.3577490

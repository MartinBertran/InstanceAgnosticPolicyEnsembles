# InstanceAgnosticPolicyEnsembles

This is an implementation of the Instance Agnostic Policy Ensemble method proposed in 

Instance-based Generalization in Reinforcement Learning

To be presented in NeurIPS 2020

Citing Instance-based Generalization in Reinforcement Learning
------------------

If you reference or use IAPE in your research, please cite:

```
@article{bertran2020instance,
  title={Shaping belief states with generative environment models for RL},
  author={Bertran, Martin and Martinez, Natalia and Phielipp, Mariano and Sapiro, Guillermo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

Instalation
------------

Prerequisites:

A Dockerfile with all required dependencies is provided

To use coinrun, install the provided coinrun environment, it adds current level to the environment info
```
cd coinrun; pip install -e .
```

To use ProcGen, you can directly install the current ProcGen repo via 
```
pip install procgen
```
or use the custom procgen environment provided, the latter has native CutOut support, which  accelerates environment interactions significantly when performing this augmentation
```
cd procgen; pip install -e .
```


Finally, install IAPE via 
```
pip install -e .
```

Running an experiment
------------

To run an experiment on the procgen version of coinrun with 10 ensemble heads and 500 training levels, simply run
```
python iape/iape/run_iape --gpu 0  --exp_name test --epochs 2000 --lr_v 2e-4 --env procgen:procgen-coinrun-v0 --n_ens 10
```
Additional options are present in the same file


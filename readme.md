

# LAIES

Open-source code for [Lazy Agents: A New Perspective on Solving Sparse Reward Problem in Multi-agent Reinforcement Learning](https://proceedings.mlr.press/v202/liu23ac.html).

The paper is  accepted by ICML 2023. Our approach can help both value-based and policy-based baselines (such as QMIX, QPLEX, IPPO, and MAPPO) to avoid lazy agent for improving learning efficiency in challenging sparse reward benchmarks.



## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=3m beta1=100 beta2=.1 label=LAIES wandb=True t_max=2500000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=1c3s5z beta1=100  beta2=2 label=LAIES itrin_two_clip=0.7 t_max=3200000 wandb=True seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=2m_vs_1z beta1=600 beta2=2 anneal_intrin=True itrin_two_clip=0.4 label=LAIES wandb=True anneal_speed=4000000 t_max=3200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=5m_vs_6m beta1=200 beta2=2 label=LAIES anneal_speed=4000000 wandb=True t_max=5200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=MMM2 beta1=100 beta2=2 label=LAIES anneal_speed=4000000 t_max=5200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=6h_vs_8z beta1=100 beta2=2 label=LAIES td_lambda=0.3 epsilon_anneal_time=500000 anneal_speed=4000000 wandb=True t_max=10200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=3s_vs_3z beta1=100 beta2=2 label=LAIES wandb=True t_max=4200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=8m_vs_9m beta1=10 beta2=20 label=LAIES itrin_two_clip=0.7  anneal_speed=2000000 wandb=True t_max=5200000 seed=125
python main.py --config=LA_SMAC --env-config=sc2 with env_args.map_name=MMM beta1=20 beta2=2 label=LAIES wandb=True anneal_speed=4000000 t_max=2200000 seed=125

python main.py --config=LA_SMAC_PPO --env-config=sc2 with env_args.map_name=3m beta1=90 beta2=0 label=LAIES wandb=True t_max=3200000 t_max=3200000 seed=125
python main.py --config=LA_SMAC_PPO --env-config=sc2 with env_args.map_name=3s_vs_3z beta1=100 beta2=0 label=LAIES anneal_speed=4000000 wandb=True t_max=4200000 seed=125
python main.py --config=LA_SMAC_PPO --env-config=sc2 with env_args.map_name=1c3s5z beta1=200 beta2=2 label=LAIES itrin_two_clip=0.7 t_max=3200000 wandb=True seed=125
python main.py --config=LA_SMAC_PPO --env-config=sc2 with env_args.map_name=MMM beta1=30 beta2=0 label=LAIES anneal_speed=4000000 t_max=3200000 seed=125
```


[//]: # (```shell)

[//]: # (# For Google Football )

[//]: # (# map_name: academy_counterattack_easy, academy_counterattack_hard, academy_3_vs_1_with_keeper...)

[//]: # (python main.py --config=LA_GRF --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper beta1=1 beta2=8 label=LAIES wandb=True t_max=10200000  seed=125)

[//]: # (python main.py --config=LA_GRF --env-config=gfootball with env_args.map_name=academy_counterattack_easy beta1=1 beta2=8 label=LAIES wandb=True t_max=10200000  seed=125)

[//]: # (python main.py --config=LA_GRF --env-config=gfootball with env_args.map_name=academy_counterattack_hard beta1=1 beta2=8 label=LAIES wandb=True t_max=10200000  seed=125)

[//]: # (```)

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

Our code uses WandB for visualization. Before you run it, please configure [WandB](https://wandb.ai/site).

**Run n parallel experiments**

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`, and we store the test wining rate with csv format in `csv_files`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

# Citation

```

@InProceedings{pmlr-v202-liu23ac,
  title = 	 {Lazy Agents: A New Perspective on Solving Sparse Reward Problem in Multi-agent Reinforcement Learning},
  author =       {Liu, Boyin and Pu, Zhiqiang and Pan, Yi and Yi, Jianqiang and Liang, Yanyan and Zhang, D.},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {21937--21950},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}

```

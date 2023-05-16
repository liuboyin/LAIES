import numpy as np
import os,random,subprocess
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    th.set_num_threads(4)
    # run
    if "use_per" in _config and _config["use_per"]:
        run_REGISTRY['per_run'](_run, config, _log)
    else:
        run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _get_free_gpu_id(free_size=0.7):
    all_command = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
    all_result = subprocess.getoutput(all_command)
    # free
    if len(all_result)>0:
        all_data = [float(item.split(':')[1].strip('MiB').strip(' ')) for item in all_result.split('\n')]
        return np.argmin(all_data)
    else: return random.randint(0,2)
def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    os.environ['SC2PATH'] = "/home/liuboyin/StarCraftII"

    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)



    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # env_config['project_name'] = 'Lazy_Avoid'
    env_config['wandb'] = True
    env_config['label'] = 'opp_explore'

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    if config_dict['env_args']['map_name']=='corridor':
        config_dict['t_max']=5.5e6
    elif config_dict['env_args']['map_name']=='MMM2':
        config_dict['t_max']=2.5e6
    elif config_dict['env_args']['map_name']=='academy_3v2' or config_dict['env_args']['map_name'] == 'academy_run_pass_and_shoot_with_keeper':
        config_dict['t_max']=4e6
    elif config_dict['env_args']['map_name'] == '3s5z_vs_3s6z' or config_dict['env_args']['map_name'] == '3s_vs_8z':
        config_dict['t_max']=6e6
    elif config_dict['env_args']['map_name'] == '3s5z_vs_3s7z':
        config_dict['t_max']=11e6
    else:
        config_dict['t_max']=11e6
    # now add all the config to sacred
    config_dict['seed'] = random.randint(1000,9999)
    config_dict['cuda_num'] = _get_free_gpu_id()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict['cuda_num'])
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()

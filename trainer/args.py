import os
import re
import torch
import importlib
import transformers
from typing import Optional
from dataclasses import field, dataclass
from .utils_hf import cached_property, logging, requires_backends


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model: str = field(
        default='swin'
    )
    dataset: str = field(
        default='cmip6', metadata={"help": "Training dataset."}
    )
    lr_scheduler_type_custom: str = field(
        default=None
    )
    min_learning_rate: float = field(
        default=0.0, metadata={"help": "The minimum of learning rate."}
    )
    dist_port: Optional[int] = field(
        default=21111, metadata={"help": "The port used for distributed training"}
    )
    inputs_key_for_metrics: Optional[str] = field(
        default=None, metadata={"help": "The key in dictionary of inputs that will be passed to the `compute_metrics` function."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.include_inputs_for_metrics and self.inputs_key_for_metrics is None:
            raise ValueError(
                "--include_inputs_for_metrics requires inputs_key_for_metrics not to be None"
            )

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        elif 'SLURM_PROCID' in os.environ:
            return int(os.environ['SLURM_NTASKS'])
        else:
            return 1
            # assert 0, "Can't get the correct world_size."

def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1, ip2, ip3, ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr

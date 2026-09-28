from .fql_config import fql_config
from .sacbc_config import sacbc_config
from .qsm_config import qsm_config
from .dsrl_config import dsrl_config
from .ifql_config import ifql_config

configs = {
    "fql": fql_config,
    "ifql": ifql_config,
    "sacbc": sacbc_config,
    "qsm": qsm_config,
    "dsrl": dsrl_config,
}

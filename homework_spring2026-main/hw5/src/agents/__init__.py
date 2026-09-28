from .fql_agent import FQLAgent
from .iql_agent import IQLAgent
from .sacbc_agent import SACBCAgent

agents = {
    "fql": FQLAgent,
    "iql": IQLAgent,
    "sacbc": SACBCAgent,
}

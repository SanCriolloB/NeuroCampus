from .strategies.rbm_manual_strategy import RBMManualStrategy
from .strategies.bm_manual_strategy import BMManualStrategy
from .strategies.dbm_manual_strategy import DBMManualStrategy

REGISTRY = {
    "rbm_manual": RBMManualStrategy,
    "bm_manual": BMManualStrategy,
    "dbm_manual": DBMManualStrategy,
}

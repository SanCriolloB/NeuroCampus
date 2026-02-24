import tempfile

import numpy as np

from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral


def test_warm_start_classification_loads_weights_when_compatible():
    """Warm-start en clasificación debe cargar pesos si task/cols/shape coinciden."""

    # 1) Crear un "modelo base" con seed=1 y guardarlo
    m1 = RBMGeneral(n_hidden=8, cd_k=1, seed=1)
    m1.setup(
        data_ref=None,
        hparams={
            "task_type": "classification",
            "seed": 1,
            "max_calif": 10,
            "use_text_embeds": False,
            "use_text_probs": False,
        },
    )

    with tempfile.TemporaryDirectory() as td:
        m1.save(td)

        # Snapshot de pesos base
        W1 = m1.rbm.W.detach().cpu().numpy().copy()
        head1 = next(iter(m1.head.parameters())).detach().cpu().numpy().copy()

        # 2) Crear un "modelo nuevo" con seed distinto (pesos distintos) pero compatible
        m2 = RBMGeneral(n_hidden=8, cd_k=1, seed=7)
        m2.setup(
            data_ref=None,
            hparams={
                "task_type": "classification",
                "seed": 7,
                "warm_start_path": td,
                "max_calif": 10,
                "use_text_embeds": False,
                "use_text_probs": False,
            },
        )

        assert isinstance(getattr(m2, "warm_start_info_", None), dict)
        assert m2.warm_start_info_.get("warm_start") == "ok"

        # 3) Verificar que los pesos fueron reemplazados por los del modelo base
        W2 = m2.rbm.W.detach().cpu().numpy()
        head2 = next(iter(m2.head.parameters())).detach().cpu().numpy()

        assert np.allclose(W1, W2)
        assert np.allclose(head1, head2)

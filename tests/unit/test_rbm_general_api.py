import numpy as np
import pandas as pd
from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral

def test_rbm_general_fit_predict_proba_and_predict():
    # DataFrame con calif_* y un target limpio
    N = 120
    df = pd.DataFrame({f"calif_{i+1}": np.random.rand(N).astype(np.float32)*5.0 for i in range(10)})
    # target: neg/neu/pos por regla simple
    y_txt = np.where(df["calif_1"].values>2.5, "pos", "neg")
    y_txt[:N//3] = "neu"
    df["sentiment_label_teacher"] = y_txt

    m = RBMGeneral(n_hidden=8, cd_k=1, seed=7)
    info = m.fit(
        df,
        scale_mode="scale_0_5",
        epochs=2, epochs_rbm=1, batch_size=32, lr_rbm=0.01, lr_head=0.01,
        use_text_probs=False, use_text_embeds=False, max_calif=10
    )
    assert "accuracy" in info and "f1_macro" in info

    proba = m.predict_proba_df(df.iloc[:10].copy())
    assert proba.shape == (10,3)
    yhat = m.predict_df(df.iloc[:10].copy())
    assert len(yhat) == 10
    assert set(yhat).issubset({"neg","neu","pos"})

def test_rbm_general_regression_predict_score_df_save_load(tmp_path):
    # Dataset tipo pair_matrix: docente-materia + features numéricas + target
    N = 80
    rng = np.random.default_rng(7)

    df = pd.DataFrame({
        "periodo": ["2025-1"] * N,
        "teacher_id": rng.integers(0, 12, size=N),
        "materia_id": rng.integers(0, 18, size=N),
        "target_score": (rng.random(N).astype(np.float32) * 50.0),
    })
    for i in range(5):
        df[f"calif_{i+1}"] = (rng.random(N).astype(np.float32) * 5.0)

    data_path = tmp_path / "pair.csv"
    df.to_csv(data_path, index=False)

    m = RBMGeneral(n_hidden=8, cd_k=1, seed=7)
    hparams = {
        "task_type": "regression",
        "target_col": "target_score",
        "include_teacher_materia": True,
        "teacher_materia_mode": "embed",
        "teacher_id_col": "teacher_id",
        "materia_id_col": "materia_id",
        "batch_size": 16,
        "epochs_rbm": 1,
        "epochs": 2,
        "lr_rbm": 0.01,
        "lr_head": 0.01,
        "val_ratio": 0.2,
        "split_mode": "random",
        "dropout": 0.0,
    }

    m.setup(str(data_path), hparams)
    # entrenar 2 épocas rápidas
    for ep in range(2):
        _ = m.train_step(ep, hparams)

    out_dir = tmp_path / "model"
    m.save(str(out_dir))

    m2 = RBMGeneral.load(str(out_dir), device="cpu")

    # inferencia sin target
    df_infer = df.drop(columns=["target_score"]).copy()
    scores = m2.predict_score_df(df_infer)

    assert scores.shape == (len(df_infer),)
    assert np.isfinite(scores).all()
    # debe estar en escala 0..50 (con clip en service, pero aquí validamos rango razonable)
    assert float(scores.min()) >= -1.0
    assert float(scores.max()) <= 60.0

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

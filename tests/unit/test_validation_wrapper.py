# tests/unit/test_validation_wrapper.py
import pandas as pd
from neurocampus.data.validation_wrapper import run_validations

def test_wrapper_ok_minimo():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    rep = run_validations(df, dataset_id="docentes")
    assert "summary" in rep and "issues" in rep
    assert rep["summary"]["rows"] == 2
    assert rep["summary"]["errors"] >= 0

def test_wrapper_dataframe_vacio():
    df = pd.DataFrame()
    rep = run_validations(df)
    assert rep["summary"]["rows"] == 0
    assert rep["summary"]["errors"] >= 1 or rep["ok"] is False

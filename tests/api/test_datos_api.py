# tests/api/test_datos_api.py

def test_validar_csv_ok(client):
    csv = "col1,col2\n1,2\n3,4\n"
    files = {"file": ("sample.csv", csv, "text/csv")}
    data = {"dataset_id": "docentes"}
    r = client.post("/datos/validar", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["ok"], bool)
    assert body.get("dataset_id") == "docentes"
    assert isinstance(body.get("sample"), list)

def test_validar_formato_no_soportado(client):
    files = {"file": ("readme.txt", "hola", "text/plain")}
    data = {"dataset_id": "docentes"}
    r = client.post("/datos/validar", files=files, data=data)
    assert r.status_code == 400

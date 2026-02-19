"""Dashboard smoke test.

Ejecuta un conjunto mínimo de requests contra la API del Dashboard y falla si
algún endpoint responde distinto de HTTP 200 o si el body no es JSON válido.

Uso
---
1) Levanta el backend:
   make be-dev

2) Ejecuta el smoke test:
   PYTHONPATH=backend/src python backend/scripts/dashboard_smoke_test.py

Opcionalmente puedes ajustar:
- Base URL: --base http://localhost:8000
- Rango de periodos: --periodo-from 2024-1 --periodo-to 2025-1
- Filtros: --docente "..." --asignatura "..." --programa "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Check:
    """Representa un check HTTP contra un endpoint."""

    name: str
    path: str
    params: Dict[str, Any]


def _build_url(base: str, path: str, params: Dict[str, Any]) -> str:
    base = base.rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    clean: Dict[str, str] = {}
    for k, v in (params or {}).items():
        if v is None:
            continue
        clean[k] = str(v)
    qs = urllib.parse.urlencode(clean)
    return f"{base}{path}" + (f"?{qs}" if qs else "")


def _http_get_json(url: str, timeout: float = 20.0) -> Tuple[int, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        return int(getattr(e, "code", 0) or 0), body
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"

    try:
        return status, json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001
        return status, f"JSONDecodeError: {e} | body={raw[:400]!r}"


def _print_fail(name: str, url: str, status: int, body: Any) -> None:
    print(f"[FAIL] {name}: {status} -> {url}")
    if isinstance(body, str):
        preview = body if len(body) <= 700 else (body[:700] + "...")
        print("       ", preview.replace("\n", " ")[:900])
    else:
        try:
            s = json.dumps(body, ensure_ascii=False)[:900]
            print("       ", s)
        except Exception:
            print("       ", repr(body)[:900])


def _checks(periodo_from: str, periodo_to: str, docente: Optional[str], asignatura: Optional[str], programa: Optional[str]) -> List[Check]:
    # Filtros comunes (solo se agregan si vienen en CLI)
    f = {
        "periodo_from": periodo_from,
        "periodo_to": periodo_to,
        "docente": docente,
        "asignatura": asignatura,
        "programa": programa,
    }

    return [
        Check("status", "/dashboard/status", {}),
        Check("periodos", "/dashboard/periodos", {}),
        Check("catalogos", "/dashboard/catalogos", f),
        Check("kpis", "/dashboard/kpis", f),
        Check("series_evaluaciones", "/dashboard/series", {**f, "metric": "evaluaciones"}),
        Check("series_score", "/dashboard/series", {**f, "metric": "score_promedio"}),
        Check("rankings_docente_best", "/dashboard/rankings", {**f, "by": "docente", "metric": "score_promedio", "order": "desc", "limit": 5}),
        Check("rankings_docente_worst", "/dashboard/rankings", {**f, "by": "docente", "metric": "score_promedio", "order": "asc", "limit": 5}),
        Check("radar", "/dashboard/radar", f),
        Check("wordcloud", "/dashboard/wordcloud", {**f, "limit": 20}),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test de endpoints /dashboard/*")
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL del backend (default: http://localhost:8000)")
    parser.add_argument("--periodo-from", default="2024-1", help="Periodo inicio (incl.) (default: 2024-1)")
    parser.add_argument("--periodo-to", default="2025-1", help="Periodo fin (incl.) (default: 2025-1)")
    parser.add_argument("--docente", default=None, help="Filtro opcional por docente (exact match)")
    parser.add_argument("--asignatura", default=None, help="Filtro opcional por asignatura (exact match)")
    parser.add_argument("--programa", default=None, help="Filtro opcional por programa (exact match)")
    args = parser.parse_args(argv)

    checks = _checks(args.periodo_from, args.periodo_to, args.docente, args.asignatura, args.programa)

    ok = 0
    fail = 0
    for c in checks:
        url = _build_url(args.base, c.path, c.params)
        status, body = _http_get_json(url)
        if status != 200:
            fail += 1
            _print_fail(c.name, url, status, body)
            continue

        # Verifica JSON estructural (no validamos schema estrictamente aquí)
        if isinstance(body, str) and body.startswith("JSONDecodeError"):
            fail += 1
            _print_fail(c.name, url, status, body)
            continue

        ok += 1
        print(f"[OK]   {c.name}: 200")

    print("\nResumen:")
    print(f"  OK  : {ok}")
    print(f"  FAIL: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

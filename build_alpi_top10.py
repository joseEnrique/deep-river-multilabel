"""Genera el top-10 de alpi para la ablación de ganancias/decay.

En alpi cada máquina es un experimento distinto (el runner instancia NewAlpi
con `machine`), así que un "modelo" son 4 runs: uno por máquina del grid de
config-alpi-bce-static-grid.yaml. Ordenar por macro-F1 de runs sueltos daría un
top-10 entero de la máquina 0 (la más fácil), así que se agrupa por
configuración sin `machine` y se ordena por la MEDIA de las 4.

Salida: alpiablation_top10_static.json, con la config base y los 4 runs
(máquina, exp_name y su macro-F1 StaticFocal) de cada modelo.
"""
from __future__ import annotations
import io, json, sys
import requests
import pandas as pd

BASE = "http://13.37.80.70:8080/api/v1/datasets/alpi"
H = {"X-API-Key": "Golaso1992?"}
MACHINES = [0, 2, 5, 18]          # las del grid bce/static
TOP_N = 10
OUT = "alpiablation_top10_static.json"


def main() -> int:
    r = requests.get(f"{BASE}/results.csv",
                     params={"status": "done", "loss_type": "StaticFocal"},
                     headers=H, timeout=600)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df = df[df["cfg.machine"].isin(MACHINES)]

    # Agrupar por todo menos la máquina: `device` y `dataset` se excluyen
    # también porque no cambian el experimento (device es dónde tocó correr).
    keys = [c for c in df.columns if c.startswith("cfg.")
            and c not in ("cfg.machine", "cfg.device", "cfg.dataset")]
    for c in ("cfg.hidden_dims", "cfg.hidden_dim"):
        df[c] = df[c].astype(str)

    g = df.groupby(keys, dropna=False)
    agg = g["metric.macro_f1"].agg(["mean", "count"])
    full = agg[agg["count"] == len(MACHINES)].sort_values("mean", ascending=False)
    print(f"{len(agg)} configs, {len(full)} con las {len(MACHINES)} máquinas")

    salida = []
    for rank, (clave, fila) in enumerate(full.head(TOP_N).iterrows(), 1):
        sub = g.get_group(clave)[["exp_name", "cfg.machine", "metric.macro_f1"]]
        sub = sub.sort_values("cfg.machine")
        # La config se toma del documento del backend, no se reconstruye desde
        # las columnas planas: así viaja tal cual la usó el runner.
        rep = sub.iloc[0]["exp_name"]
        doc = requests.get(f"{BASE}/experiments/{rep}", headers=H, timeout=60).json()
        cfg = {k: v for k, v in doc["config"].items() if k != "machine"}
        runs = [{"machine": int(m), "exp_name": n, "macro_f1": float(f)}
                for n, m, f in sub.itertuples(index=False)]
        salida.append({
            "architecture": doc.get("architecture") or cfg["architecture"],
            "mean_macro_f1": float(fila["mean"]),
            "config": cfg,
            "runs": runs,
        })
        print(f"{rank:2} media {fila['mean']:6.2f}  " +
              "  ".join(f"m{r['machine']}={r['macro_f1']:.2f}" for r in runs))

    with open(OUT, "w") as f:
        json.dump({"alpi": salida}, f, indent=1)
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

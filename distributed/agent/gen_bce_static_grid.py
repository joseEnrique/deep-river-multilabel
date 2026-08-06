"""Genera los 3 configs (AI4I, NPS, ALPI) del grid COMPLETO SIN adaptive:
solo BCE + StaticFocal(alpha=0.25, gamma=2.0).

Grid = factorial uniforme de analysis/grid_tablas.tex, COMPLETADO (hidden 32/64/128 y
lr 1e-4/1e-3/1e-2 para las 3 backbones secuenciales; MLP con sus stacks).
  - AI4I / NPS: mono-maquina, RB {1,50,200,500}.
  - ALPI: RB {1,50,200,500}; se ejecuta solo en el subconjunto de maquinas
          que reproduce la media de las 20.
Registro idempotente en backend (INSERT OR IGNORE por exp_name): re-emitir lo ya hecho es inocuo.

Salida: config-{ai4i,nps,alpi}-bce-static-grid.yaml (los consume el agente distribuido).
"""
import itertools, yaml


class NoAliasDumper(yaml.SafeDumper):
    """Evita anchors/aliases (*id00x): escribe cada lista expandida, no por referencia."""
    def ignore_aliases(self, data):
        return True

# ── PARAMETROS QUE PUEDES TOCAR ───────────────────────────────────────────────
ALPI_MACHINES = [0, 2, 5, 18]  # subconjunto que reproduce la media de las 20 (MAE 0.91)
                               #   alternativas: [0,2,18] (0.89) | [0,1,2,5,11,13,17,18] (0.80)
LOSSES = [
    {"type": "BCE"},
    {"type": "StaticFocal", "alpha": 0.25, "gamma": 2.0},
]
# ──────────────────────────────────────────────────────────────────────────────

# Ejes comunes del grid (factorial uniforme)
DROPOUT = [0.0, 0.2]
NORM    = ["none", "layernorm"]
LR      = [0.0001, 0.001, 0.01]
PH      = [1, 2, 5, 10, 20]
EPOCHS  = [1, 3, 10, 20]
HIDDEN  = [32, 64, 128]                              # CNN / LSTM / Transformer
DEPTH   = [1, 2]
MLP_STACKS = [[64, 32], [128, 64], [128, 64, 32]]    # MLP: el stack define la profundidad

DEVICE = {"CNN": "cuda:0", "LSTM": "cuda:0", "Transformer": "cuda:1", "MLP": "cuda:1"}

# Especificacion por dataset
DATASETS = {
    "ai4i": dict(rb=[1, 50, 200, 500], machines=None, alpi=False),
    "nps":  dict(rb=[1, 50, 200, 500], machines=None, alpi=False),
    "alpi": dict(rb=[1, 50, 200, 500], machines=ALPI_MACHINES, alpi=True),
}


def common_axes(spec):
    ax = dict(optimizer=["adam"], seed=[42],
              past_history=PH, window_size=spec["rb"], epochs=EPOCHS,
              dropout=DROPOUT, normalization=NORM, lr=LR)
    if spec["alpi"]:
        ax.update(input_win=[1720], output_win=[480], delta=[0], sigma=[120],
                  min_count=[0], num_alarms=[155], embedding_dim=[8],
                  machine=spec["machines"])
    return ax


def seq_block(arch, spec):
    b = {"architecture": [arch], "device": [DEVICE[arch]],
         "hidden_dim": HIDDEN, "num_layers": DEPTH, "bidirectional": [False]}
    b.update(common_axes(spec))
    return b


def mlp_block(spec):
    b = {"architecture": ["MLP"], "device": [DEVICE["MLP"]],
         "hidden_dims": [list(s) for s in MLP_STACKS]}
    b.update(common_axes(spec))
    return b


def build_model(spec):
    return [seq_block("CNN", spec), seq_block("LSTM", spec),
            seq_block("Transformer", spec), mlp_block(spec)]


def expand(block):
    keys = list(block.keys())
    vals = [block[k] if isinstance(block[k], list) else [block[k]] for k in keys]
    return sum(1 for _ in itertools.product(*vals))


TOPLEVEL = dict(
    backend_url="http://13.37.80.70:8080", api_key="Golaso1992?", agent_id=None,
    devices=["cuda:0", "cuda:1"], max_slots_per_device=4,
    slots_per_device={"cuda:0": 4, "cuda:1": 2}, consume_any=True,
    checkpoint_every=1000, poll_interval=5,
)

grand = 0
for ds, spec in DATASETS.items():
    model = build_model(spec)
    top = dict(TOPLEVEL); top["dataset"] = ds
    out = f"config-{ds}-bce-static-grid.yaml"
    with open(out, "w") as f:
        f.write(f"# Grid COMPLETO {ds.upper()} (factorial uniforme) x {{BCE, StaticFocal a0.25 g2.0}} -- SIN adaptive\n")
        f.write(f"# RB={spec['rb']}"
                + (f"  maquinas={spec['machines']} (subconjunto que reproduce la media de las 20)\n"
                   if spec["alpi"] else "  (mono-maquina)\n"))
        f.write("# Idempotente en backend (INSERT OR IGNORE por exp_name).\n")
        # mismo orden que config.example.yaml: top-level, luego model, y loss AL FINAL.
        yaml.dump(top, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
        yaml.dump({"model": model}, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=None, width=120)
        f.write("\n")
        yaml.dump({"loss": LOSSES}, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

    mm = sum(expand(b) for b in model)          # incluye machine si aplica
    nmach = len(spec["machines"]) if spec["alpi"] else 1
    configs = mm // nmach                        # configs de modelo (sin machine, 1 loss)
    total = mm * len(LOSSES)
    grand += total
    print(f"[{ds}] {out}")
    print(f"   RB={spec['rb']} machines={spec['machines']} | "
          f"configs_modelo={configs} | model x machine={mm} | TOTAL x{len(LOSSES)} losses = {total}")

print(f"\nGRAN TOTAL (3 datasets, 2 losses): {grand} experimentos")

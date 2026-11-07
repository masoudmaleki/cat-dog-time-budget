"""R03 E: translating harness features into collar axis meaning (task 03 definition).
Harness: X cranio-caudal, Y dorso-ventral, Z lateral. Collar: X lateral, Y dorso-ventral, Z cranio-caudal.
Mapping: collar_X = harness_Z ; collar_Z = harness_X ; collar_Y = −harness_Y (R01 posture signature: harness Y standing +0.79, collar Y −0.38).
"""
import numpy as np, pandas as pd
from src import smit_data as D

STATS = ("Mean", "Min", "Max", "Sum", "sd", "Skew", "Kurt")

# Table: collar column -> (harness column, sign). For Y, Min/Max are swapped.
MAP = {}
for s in STATS:
    MAP[f"X_{s}"] = (f"Z_{s}", +1)
    MAP[f"Z_{s}"] = (f"X_{s}", +1)
MAP.update({"Y_Mean": ("Y_Mean", -1), "Y_Sum": ("Y_Sum", -1), "Y_Min": ("Y_Max", -1), "Y_Max": ("Y_Min", -1),
            "Y_Skew": ("Y_Skew", -1), "Y_sd": ("Y_sd", +1), "Y_Kurt": ("Y_Kurt", +1)})
for s in STATS:
    MAP[f"VM_{s}"] = (f"VM_{s}", +1)
MAP["ODBA30_Mean"] = ("ODBA30_Mean", +1)
MAP.update({"Cor_XY": ("Cor_YZ", -1), "Cor_YZ": ("Cor_XY", -1), "Cor_XZ": ("Cor_XZ", +1)})

INV_COLS = [f"VM_{s}" for s in STATS] + ["ODBA30_Mean", "X_sd", "Y_sd", "Z_sd", "X_Kurt", "Y_Kurt", "Z_Kurt"]   # orientation-insensitive (14)


def harness_to_collar(XH):
    """XH: DataFrame (at least the FEAT32 columns). Returns: harness features mapped to collar-named columns (in FEAT32 order)."""
    out = {}
    for col in D.FEAT32:
        src, sg = MAP[col]
        out[col] = sg * XH[src].values
    return pd.DataFrame(out, index=XH.index)


def table_rows():
    return [dict(tasma_sutun=k, kosum_kaynak=v[0], isaret="−" if v[1] < 0 else "+") for k, v in MAP.items()]

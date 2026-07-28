"""
Build a compact Table 4 for manuscript use.

Changes versus the raw component-endpoint table:
- Drop N
- Merge Events and Event rate into one column
- Merge each model's C-index and 95% CI into one column
- Export a Word-friendly RTF plus CSV/PDF/PNG previews
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
SRC = ROOT / "results/manuscript_final/main_tables/component_endpoint_cindex.csv"
OUT_TABLE = ROOT / "results/manuscript_final/main_tables/component_endpoint_cindex_compact.csv"
OUT_RTF = ROOT / "manuscript/table4_compact_landscape.rtf"
OUT_PNG = ROOT / "manuscript/table4_compact_landscape.png"
OUT_PDF = ROOT / "manuscript/table4_compact_landscape.pdf"


def build_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    compact = pd.DataFrame()
    compact["Endpoint"] = df["Endpoint"].replace({
        "CV death": "CV death",
        "MI": "MI",
        "HF": "HF",
        "Stroke": "Stroke",
        "CVD_Composite_4": "CVD composite 4",
    })
    compact["Events, n (%)"] = (
        df["Events"].astype(str) + " (" + df["Event rate"].astype(str) + ")"
    )

    for i in range(6):
        c_col = f"Model {i} C" if i < 5 else "Model 5 (DL-ECG) C"
        ci_col = f"Model {i} 95CI" if i < 5 else "Model 5 (DL-ECG) 95CI"
        out_col = f"Model {i}" if i < 5 else "Model 5 (DL-ECG)"
        compact[out_col] = df[c_col].astype(str) + " " + df[ci_col].astype(str)

    return compact


def esc_rtf(text: str) -> str:
    s = str(text).replace("—", "-").replace("±", "+/-")
    out = []
    for ch in s:
        code = ord(ch)
        if ch == "\\":
            out.append(r"\\")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif code > 127:
            out.append(rf"\u{code}?")
        else:
            out.append(ch)
    return "".join(out)


def write_rtf(df: pd.DataFrame, out_path: Path) -> None:
    widths = [1700, 1450, 1500, 1500, 1500, 1500, 1500, 1650]
    cellxs = []
    cur = 0
    for w in widths:
        cur += w
        cellxs.append(cur)

    def row(values, header=False, fs=16):
        parts = [r"\trowd\trgaph40\trleft0"]
        if header:
            parts.append(r"\trhdr")
        for x in cellxs:
            parts.append(rf"\cellx{x}")
        body = []
        for val in values:
            prefix = r"\intbl "
            if header:
                prefix += r"\b "
            prefix += rf"\fs{fs} "
            body.append(prefix + esc_rtf(val) + r"\cell")
        return "".join(parts) + "\n" + "\n".join(body) + "\n" + r"\row" + "\n"

    rtf = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Arial;}}",
        r"\viewkind4\uc1",
        r"\sectd\lndscpsxn\pgwsxn16840\pghsxn11907\marglsxn540\margrsxn540\margtsxn540\margbsxn540",
        (
            r"\pard\sa180\qc\b\fs22 "
            + esc_rtf(
                "Table 4. Component-endpoint discrimination of Models 0-5 for "
                "cardiovascular death, myocardial infarction, heart failure "
                "hospitalization, stroke, and the primary composite endpoint."
            )
            + r"\b0\par"
        ),
        r"\pard\sa120\par",
        row(df.columns.tolist(), header=True, fs=15),
    ]

    for _, rec in df.iterrows():
        rtf.append(row(rec.tolist(), header=False, fs=15))

    rtf.append("}")
    out_path.write_text("\n".join(rtf), encoding="utf-8")


def write_preview(df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 2.5), dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.02, 1.75)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")

    fig.text(
        0.5,
        0.98,
        (
            "Table 4. Component-endpoint discrimination of Models 0-5 for "
            "cardiovascular death, myocardial infarction, heart failure "
            "hospitalization, stroke, and the primary composite endpoint."
        ),
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    plt.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    df = pd.read_csv(SRC)
    compact = build_compact_table(df)
    compact.to_csv(OUT_TABLE, index=False)
    write_rtf(compact, OUT_RTF)
    write_preview(compact, OUT_PNG, OUT_PDF)

    print(f"Saved: {OUT_TABLE}")
    print(f"Saved: {OUT_RTF}")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()

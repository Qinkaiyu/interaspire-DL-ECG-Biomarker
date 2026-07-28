"""
Merge main-manuscript figures to reduce figure count without dropping content.

Outputs
-------
1. Figure 3 merged (1×3 row, equal panel size):
   - Panel A: overall KM
   - Panels 2–3: normal / abnormal ECG KM (one group, label B only)

2. Figure 5 merged:
   - Panel A: lead-wise averaged waveforms
   - Panel B: PheWAS
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


BASE = Path(".")
FIG_DIR = BASE / "results" / "manuscript_final" / "main_figures"
WAVE_DIR = BASE / "results" / "manuscript_final" / "ecg_waveform_cvd4"


def trim_white(img: Image.Image, pad: int = 12) -> Image.Image:
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img.convert("RGB"), bg.convert("RGB"))
    bbox = diff.getbbox()
    if bbox is None:
        return img
    left = max(bbox[0] - pad, 0)
    upper = max(bbox[1] - pad, 0)
    right = min(bbox[2] + pad, img.size[0])
    lower = min(bbox[3] + pad, img.size[1])
    return img.crop((left, upper, right, lower))


def load_trimmed(path: Path) -> Image.Image:
    return trim_white(Image.open(path))


def pad_to_same_size(imgs):
    """Center each image on a common canvas so three panels share identical pixel size."""
    tw = max(im.width for im in imgs)
    th = max(im.height for im in imgs)
    out = []
    for im in imgs:
        canvas = Image.new("RGB", (tw, th), "white")
        src = im.convert("RGB") if im.mode != "RGB" else im
        x = (tw - src.width) // 2
        y = (th - src.height) // 2
        canvas.paste(src, (x, y))
        out.append(canvas)
    return out


def merged_figure3() -> None:
    overall = load_trimmed(FIG_DIR / "fig3_km_quartile_CVD4.png")
    ecg_status = load_trimmed(FIG_DIR / "fig4_km_normal_ecg_CVD4.png")
    w, h = ecg_status.size
    mid = w // 2
    ecg_normal = ecg_status.crop((0, 0, mid, h))
    ecg_abnormal = ecg_status.crop((mid, 0, w, h))

    imgs = pad_to_same_size([overall, ecg_normal, ecg_abnormal])

    fig = plt.figure(figsize=(20.4, 9.2), facecolor="white")
    # wspace must be >= 0 in GridSpec (negative values are ignored / clamped).
    wspace_a_vs_bc = 0.05
    wspace_b_pair = 0.0
    gs_outer = fig.add_gridspec(
        1, 2, width_ratios=[1, 2], wspace=wspace_a_vs_bc
    )
    ax_a = fig.add_subplot(gs_outer[0, 0])
    gs_b = gs_outer[0, 1].subgridspec(1, 2, wspace=wspace_b_pair)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    axes = [ax_a, ax_b1, ax_b2]

    panel_label_fs = 22
    for ax, img in zip(axes, imgs):
        ax.imshow(img, aspect="auto")
        ax.axis("off")

    fig.subplots_adjust(left=0.006, right=0.994, top=0.99, bottom=0.01)

    # Pull B1/B2: GridSpec wspace=0 时实测 gap 常为 0 → 只算 gap 时 pull=0，看起来「毫无变化」。
    # 必须加 B_PAIR_EXTRA_OVERLAP（整图宽度比例）才能压住中间白条/进一步贴紧。
    B_PAIR_CLOSE_FRAC = 1.0   # 吃掉已有缝隙的比例（对 max(gap, 0)）
    B_PAIR_EXTRA_OVERLAP = 0.02  # 与 gap 无关的额外拉近；想更紧可调大（如 0.02）
    B_PAIR_PULL_MAX = 0.2     # 单段接缝总拉近上限（过大可能与 A 重叠）
    fig.canvas.draw()
    bb1 = ax_b1.get_position()
    bb2 = ax_b2.get_position()
    gap = max(bb2.x0 - bb1.x1, 0.0)
    pull = gap * B_PAIR_CLOSE_FRAC + B_PAIR_EXTRA_OVERLAP
    pull = min(pull, B_PAIR_PULL_MAX)
    ax_b1.set_position([bb1.x0, bb1.y0, bb1.width + pull / 2, bb1.height])
    ax_b2.set_position([bb2.x0 - pull / 2, bb2.y0, bb2.width + pull / 2, bb2.height])

    # Same corner coords as panel A so "A" and "B" align; B only on first ECG tile.
    label_kw = dict(
        fontsize=panel_label_fs,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes, **label_kw)
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes, **label_kw)

    out_png = FIG_DIR / "fig3_merged_km_ecgstatus_CVD1.png"
    out_pdf = FIG_DIR / "fig3_merged_km_ecgstatus_CVD1.pdf"
    _save_tight = dict(dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    fig.savefig(out_png, **_save_tight)
    fig.savefig(out_pdf, **_save_tight)
    plt.close(fig)
    print(f"Saved {out_png}")


def merged_figure5() -> None:
    wave = load_trimmed(WAVE_DIR / "ecg_waveform_12lead_risk_groups.png")
    phewas = load_trimmed(FIG_DIR / "fig7_phewas_full_CVD4.png")

    fig = plt.figure(figsize=(16, 11), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.0], hspace=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.imshow(wave)
    ax1.axis("off")
    ax1.text(0.0, 1.02, "A", transform=ax1.transAxes, fontsize=16, fontweight="bold",
             ha="left", va="bottom")

    ax2.imshow(phewas)
    ax2.axis("off")
    ax2.text(0.0, 0.995, "B", transform=ax2.transAxes, fontsize=16, fontweight="bold",
             ha="left", va="bottom")

    out_png = FIG_DIR / "fig5_merged_waveform_phewas_CVD1.png"
    out_pdf = FIG_DIR / "fig5_merged_waveform_phewas_CVD1.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    merged_figure3()
    merged_figure5()

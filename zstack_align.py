
from pathlib import Path
from tifffile import tifffile as tf
import numpy as np
from numpy.typing import NDArray
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift
import matplotlib.pyplot as plt
import pandas as pd


# load files

def load_stack(path, channel = 0, n_channels = 2) -> NDArray:
    """
    gets a path to a specific stack file, returns a numpy array with the stack data
    """
    with tf.TiffFile(str(path)) as tif:
        n_pages = len(tif.pages)
        if n_pages % n_channels != 0:
            raise ValueError(f"{path.name}: {n_pages} pages not divisible by {n_channels}")
        data = np.stack([tif.pages[i].asarray()
                         for i in range(channel, n_pages, n_channels)])
    return data


def load_all_stacks(input_dir):
    """
    loads every .tif/.tiff in the folder, returns (names, stacks)
    """
    tif_files = sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not tif_files:
        raise FileNotFoundError(f"There are no .tif files in {input_dir}")
    names = [p.stem for p in tif_files]

    stacks = [load_stack(p) for p in tif_files]
    return names, stacks



# create projections

def choose_reference(stacks: list[NDArray]) -> int:
    """
    returns the index of the middle stack (assumes the input stacks are time-sorted)
    """
    if not stacks:
        raise ValueError("No stacks provided")
    return len(stacks) // 2


def normalize(img: NDArray, p_low = 1.0, p_high = 99.0) -> NDArray:
    """
    Gets a low percentile and a high percentile.
    Calculates the respective values in the image,
    then maps that intensity range linearly to [0, 1]
    """
    img = img.astype(np.float32)
    low_val, high_val = np.percentile(img, [p_low, p_high])
    if high_val <= low_val:
        return np.zeros_like(img)
    return np.clip((img - low_val) / (high_val - low_val), 0.0, 1.0)


def make_projections(stack: NDArray) -> dict[str, NDArray]:
    """
    Gets a 3D stack, and creates all 3 orthogonal 2D projections of it.
    For each axis, collapses the stack along that axis using max-projection,
    then normalizes the result to [0, 1]
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X), got shape {stack.shape}")
    return {
        "xy": normalize(stack.max(axis=0)),
        "xz": normalize(stack.max(axis=1)),
        "yz": normalize(stack.max(axis=2))
    }



# fitness

def ncc(a: NDArray, b: NDArray) -> float:
    """
    calculates normalized Cross-Correlation, as a fitness metrics
    range [-1, 1], 1 means a perfect match
    """
    a = a.astype(np.float32).ravel(); a -= a.mean()
    b = b.astype(np.float32).ravel(); b -= b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def fitness(ref_proj: dict[str, NDArray], moving_stack: NDArray) -> float:
    """
    calculates the mean NCC across all 3 projections
    of the moving stack vs the reference stack
    """
    mov_proj = make_projections(moving_stack)
    return float(np.mean([ncc(ref_proj[k], mov_proj[k]) for k in ("xy", "xz", "yz")]))



# alignment

def register_2d(ref_2d: NDArray, mov_2d: NDArray) -> tuple[tuple[float, float], float]:
    """
    Estimate the 2D translation that aligns mov_2d to ref_2d using phase cross-correlation.
    Uses 10x upsampling for sub-pixel precision.
    Returns ((shift_axis0, shift_axis1), confidence), where the confidence is in (0, 1]
    higher means a sharper, more reliable correlation peak.
    """
    shift_vec, error, _ = phase_cross_correlation(ref_2d, mov_2d, upsample_factor=10)
    return tuple(shift_vec), 1.0 / (1.0 + float(error))


def fuse(pairs: list[tuple[float, float]]) -> float:
    """
    Confidence-weighted average of (value, confidence) estimates.
    Used to combine the two projection-derived estimates of each translation component
    (e.g. dz from XZ and YZ) into a single value, with more weight on the higher-confidence one.
    Returns 0.0 if all confidences are zero.
    """
    vals  = np.array([v for v, _ in pairs])
    confs = np.array([c for _, c in pairs])
    return float((vals * confs).sum() / confs.sum()) if confs.sum() > 0 else 0.0


def apply_shift(stack: NDArray, shift_zyx: tuple[float, float, float]) -> NDArray:
    """
    Apply a 3D translation to a (Z, Y, X) stack
    """
    return nd_shift(stack, shift_zyx, order=1, mode="constant", cval=0.0)


def align_stack(ref_proj: dict[str, NDArray], moving_stack: NDArray) -> dict:
    """
    Register each moving_stack to the reference stack,
    via the 3 projections with confidence-weighted fusion.
    Falls back to no-op if alignment doesn't beat the unaligned baseline.
    """
    mov_proj = make_projections(moving_stack)
    (dy_xy, dx_xy), c_xy = register_2d(ref_proj["xy"], mov_proj["xy"])
    (dz_xz, dx_xz), c_xz = register_2d(ref_proj["xz"], mov_proj["xz"])
    (dz_yz, dy_yz), c_yz = register_2d(ref_proj["yz"], mov_proj["yz"])

    tx = [(dx_xy, c_xy), (dx_xz, c_xz)]
    ty = [(dy_xy, c_xy), (dy_yz, c_yz)]
    tz = [(dz_xz, c_xz), (dz_yz, c_yz)]
    shift_zyx = (fuse(tz), fuse(ty), fuse(tx))

    baseline = fitness(ref_proj, moving_stack)
    after = fitness(ref_proj, apply_shift(moving_stack, shift_zyx))
    if after <= baseline:
        shift_zyx, after = (0.0, 0.0, 0.0), baseline

    return {"shift": shift_zyx, "fitness": after,
            "baseline": baseline, "improved": after > baseline}



# results

def results_to_df(names: list[str], results: list[dict | None]) -> pd.DataFrame:
    """
    creates a df, with one row per stack
    with the columns: idx, name, baseline, fitness, delta, shift, improved
    """
    rows = []
    for i, (name, r) in enumerate(zip(names, results)):
        if r is None:
            rows.append(dict(idx=i, name=name, is_ref=True,
                             baseline=1.0, fitness=1.0, delta=0.0,
                             dz=0.0, dy=0.0, dx=0.0, improved=False))
        else:
            dz, dy, dx = r["shift"]
            rows.append(dict(idx=i, name=name, is_ref=False,
                             baseline=r["baseline"], fitness=r["fitness"],
                             delta=r["fitness"] - r["baseline"],
                             dz=dz, dy=dy, dx=dx, improved=r["improved"]))
    return pd.DataFrame(rows)



def plot_results(names, results, save_path: Path | None = None):
    """
    creates and shows a plot of before vs after alignment per stack
    and the overall mean
    """
    movers = [r for r in results if r is not None]
    befores = [r["baseline"] if r else 1.0 for r in results] + \
              [float(np.mean([r["baseline"] for r in movers]))]
    afters = [r["fitness"] if r else 1.0 for r in results] + \
             [float(np.mean([r["fitness"] for r in movers]))]
    labels  = list(names) + ["OVERALL"]
    x = list(range(len(labels)))

    plt.figure(figsize=(10, 5))
    for xi, b, a in zip(x, befores, afters):
        plt.plot([xi, xi], [b, a], color="gray", alpha=0.4)        # before→after line
    plt.scatter(x, befores, color="tab:red",   label="before", zorder=3)
    plt.scatter(x, afters,  color="tab:green", label="after",  zorder=3)
    plt.axvline(len(names) - 0.5, color="black", linestyle=":", alpha=0.4)
    plt.xticks(x, [n[-15:] for n in labels], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Fitness (NCC)")
    plt.title("Alignment quality: before vs after")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def save_stack(path: Path, stack: NDArray) -> None:
    """
    writes a 3d stack as a multi-page .tiff
    """
    tf.imwrite(str(path), stack)


def save_aligned(stacks: list[NDArray], names: list[str],
                 results: list[dict | None], output_dir: Path) -> None:
    """
    applies each best shift and writes aligned stacks to disk
    """
    for stk, name, r in zip(stacks, names, results):
        aligned = stk if r is None else apply_shift(stk, r["shift"])
        save_stack(output_dir / f"{name}_aligned.tif", aligned)
    print(f"Saved {len(stacks)} aligned stacks to {output_dir}")

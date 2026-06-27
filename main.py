

from pathlib import Path
import zstack_align as zsa


INPUT_FOLDER = "data"
OUTPUT_FOLDER = "results"

if __name__ == "__main__":
    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else input_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print('loading stacks...')
    file_names, stacks = zsa.load_all_stacks(input_dir)

    print('preparing reference stack...')
    ref_idx = zsa.choose_reference(stacks)
    ref_stack = stacks[ref_idx]
    ref_proj = zsa.make_projections(ref_stack)

    print('aligning all stacks...')
    results = [None if i == ref_idx else zsa.align_stack(ref_proj, stk)
               for i, stk in enumerate(stacks)]

    df = zsa.results_to_df(file_names, results)
    df.to_csv(output_dir / "stack_shifts.csv", index=False, float_format="%.4f")
    print(f"Wrote results to {output_dir / 'stack_shifts.csv'}")

    zsa.plot_results(file_names, results, save_path=output_dir / "alignment_plot.png")
    zsa.save_aligned(stacks, file_names, results, output_dir)
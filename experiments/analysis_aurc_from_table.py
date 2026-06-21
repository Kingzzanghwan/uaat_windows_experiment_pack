import argparse, sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# numpy >= 2.0 removed np.trapz in favor of np.trapezoid.
# This keeps the script working on both old and new numpy.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_png', required=True)
    ap.add_argument('--min_cov', type=float, default=None)
    ap.add_argument('--max_cov', type=float, default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    if 'policy' not in df.columns or 'coverage' not in df.columns or 'risk' not in df.columns:
        raise SystemExit('CSV must contain policy, coverage, risk columns')

    # Build one sorted (coverage, risk) curve per policy first, using ALL of
    # that policy's points (not yet clipped to --min_cov/--max_cov), so we
    # know the true range each policy can be safely interpolated over
    # without extrapolating.
    curves = {}
    for pol, sub in df.groupby('policy'):
        sub = sub.copy().dropna(subset=['coverage', 'risk']).sort_values('coverage')
        sub = sub.drop_duplicates(subset='coverage')
        if len(sub) < 2:
            print(f'skip {pol}: fewer than 2 usable (coverage, risk) points')
            continue
        curves[pol] = (sub['coverage'].astype(float).values, sub['risk'].astype(float).values)

    if len(curves) < 2:
        raise SystemExit('Need at least 2 policies with >=2 points to compare AURC')

    # Fair common range = the overlap of every policy's own achieved-coverage
    # range, further narrowed by --min_cov/--max_cov if given. Integrating
    # each policy over ITS OWN min/max (the old behavior) silently compares
    # different x-ranges across policies, which biases AURC toward whichever
    # policy happens to achieve lower/narrower coverage.
    common_min = max(x.min() for x, y in curves.values())
    common_max = min(x.max() for x, y in curves.values())
    if a.min_cov is not None:
        common_min = max(common_min, a.min_cov)
    if a.max_cov is not None:
        common_max = min(common_max, a.max_cov)
    if common_max <= common_min:
        raise SystemExit(
            f'No common coverage range across policies after applying --min_cov/--max_cov '
            f'(got [{common_min:.4f}, {common_max:.4f}]). Policies achieve non-overlapping '
            f'coverage ranges; AURC cannot be fairly compared as configured.'
        )

    print(f'Common coverage range used for all policies: [{common_min:.4f}, {common_max:.4f}]')
    grid = np.linspace(common_min, common_max, 200)

    rows = []
    plt.figure(figsize=(5.0, 3.4))
    for pol, (x, y) in curves.items():
        y_grid = np.interp(grid, x, y)
        area = float(_trapz(y_grid, grid))
        rows.append({
            'policy': pol,
            'coverage_min_own': float(x.min()),
            'coverage_max_own': float(x.max()),
            'common_coverage_min': float(common_min),
            'common_coverage_max': float(common_max),
            'aurc_operating_range': area,
            'n_points_own': len(x),
        })
        plt.plot(grid, y_grid, marker=None, label=f'{pol} ({area:.4f})')
        plt.plot(x[(x >= common_min) & (x <= common_max)],
                  y[(x >= common_min) & (x <= common_max)],
                  linestyle='None', marker='o', markersize=4)
    plt.xlabel('Coverage')
    plt.ylabel('Cost-sensitive risk')
    plt.title(f'Risk-Coverage Curve (common range {common_min:.2f}-{common_max:.2f})')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(a.out_png, dpi=180)

    out = pd.DataFrame(rows).sort_values('aurc_operating_range')
    out.to_csv(a.out_csv, index=False, encoding='utf-8-sig')
    print('saved:', a.out_csv, a.out_png)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()

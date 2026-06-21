import argparse, glob, os, numpy as np, pandas as pd
try:
    from scipy import stats
except Exception:
    stats = None

def load_runs(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, 'seed_*'))):
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        try:
            seed = int(os.path.basename(d).split('_')[-1])
        except Exception:
            seed = os.path.basename(d)
        m = pd.read_csv(f)
        for _, r in m.iterrows():
            rows.append({'seed': seed, 'policy': r['policy'], 'risk': r['risk'], 'coverage': r.get('coverage', np.nan), 'wrong_auto_rate': r.get('wrong_auto_rate', np.nan)})
    return pd.DataFrame(rows)

def boot_ci(x, n=5000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(bs, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(x)), float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--base', default='fixed')
    ap.add_argument('--prop', default='uaat_monotone')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = load_runs(a.root)
    if len(df) == 0:
        raise SystemExit('No seed_*/metrics.csv files found')
    rows = []
    for pol in sorted(df['policy'].unique()):
        r = df[df['policy']==pol]['risk'].astype(float).values
        m, lo, hi = boot_ci(r)
        rows.append({'policy': pol, 'n': len(r), 'mean_risk': m, 'std_risk': float(np.std(r, ddof=1)) if len(r)>1 else 0, 'ci95_lo': lo, 'ci95_hi': hi})
    piv = df.pivot_table(index='seed', columns='policy', values='risk', aggfunc='mean')
    p_t = np.nan; p_w = np.nan; mean_diff = np.nan
    if a.base in piv.columns and a.prop in piv.columns:
        pair = piv[[a.base, a.prop]].dropna()
        diff = pair[a.base].astype(float).values - pair[a.prop].astype(float).values
        mean_diff = float(np.mean(diff)) if len(diff) else np.nan
        if stats is not None and len(diff) >= 2:
            p_t = float(stats.ttest_rel(pair[a.base], pair[a.prop]).pvalue)
            if len(diff) >= 6:
                p_w = float(stats.wilcoxon(diff).pvalue)
    out = pd.DataFrame(rows)
    out['mean_diff_base_minus_prop'] = mean_diff
    out['paired_t_p'] = p_t
    out['wilcoxon_p'] = p_w
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print('saved:', a.out)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()

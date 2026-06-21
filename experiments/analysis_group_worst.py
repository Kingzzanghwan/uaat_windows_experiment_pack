import argparse, glob, os, pandas as pd

def normalize_group_file(path):
    df = pd.read_csv(path)
    pol = None
    name = os.path.basename(path)
    if 'policy' not in df.columns:
        if 'uaat' in name: pol = 'uaat_monotone'
        elif 'uncertainty' in name: pol = 'uncertainty_grid'
        elif 'fixed' in name: pol = 'fixed'
        else: pol = name.replace('group_metrics_', '').replace('.csv', '')
        df['policy'] = pol
    if 'group' not in df.columns:
        for cand in ['category','severity','wilds_split','split']:
            if cand in df.columns:
                df = df.rename(columns={cand:'group'})
                break
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True, help='group_metrics_*.csv files or glob patterns')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    paths = []
    for f in a.files:
        paths.extend(glob.glob(f))
    frames = [normalize_group_file(p) for p in paths if os.path.exists(p)]
    if not frames:
        raise SystemExit('No group metrics files found')
    g = pd.concat(frames, ignore_index=True)
    if 'group' not in g.columns or 'risk' not in g.columns:
        raise SystemExit('group metrics must contain group/category/severity/wilds_split and risk')
    rows = []
    for pol, sub in g.groupby('policy'):
        idx = sub['risk'].astype(float).idxmax()
        rows.append({'policy': pol, 'mean_group_risk': float(sub['risk'].mean()), 'median_group_risk': float(sub['risk'].median()), 'worst_group_risk': float(sub.loc[idx,'risk']), 'worst_group': str(sub.loc[idx,'group']), 'n_groups': sub['group'].nunique()})
    out = pd.DataFrame(rows).sort_values('worst_group_risk')
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print('saved:', a.out)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()

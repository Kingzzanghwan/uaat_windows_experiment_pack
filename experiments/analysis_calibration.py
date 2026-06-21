import argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ece(conf, correct, n_bins=15):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0; mce = 0.0; rows = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = float(correct[mask].mean())
        cavg = float(conf[mask].mean())
        gap = abs(acc - cavg)
        e += (mask.sum() / n) * gap
        mce = max(mce, gap)
        rows.append({'bin_lo':lo, 'bin_hi':hi, 'n':int(mask.sum()), 'avg_conf':cavg, 'accuracy':acc, 'gap':gap})
    return float(e), float(mce), pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--conf_col', default='score')
    ap.add_argument('--error_col', default='error')
    ap.add_argument('--group_col', default=None)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_png', required=True)
    ap.add_argument('--n_bins', type=int, default=15)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    conf = df[a.conf_col].astype(float).values
    # If score is not in 0-1, min-max normalize for diagnostic plot only.
    if conf.min() < 0 or conf.max() > 1:
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
    correct = 1 - df[a.error_col].astype(float).values
    rows = []
    e, m, bin_df = ece(conf, correct, a.n_bins)
    rows.append({'group':'ALL', 'ece': e, 'mce': m, 'n': len(df)})
    if a.group_col and a.group_col in df.columns:
        for name, sub in df.groupby(a.group_col):
            c = sub[a.conf_col].astype(float).values
            if c.min() < 0 or c.max() > 1:
                c = (c - c.min()) / (c.max() - c.min() + 1e-12)
            corr = 1 - sub[a.error_col].astype(float).values
            ee, mm, _ = ece(c, corr, a.n_bins)
            rows.append({'group': str(name), 'ece': ee, 'mce': mm, 'n': len(sub)})
    out = pd.DataFrame(rows).sort_values('ece', ascending=False)
    out.to_csv(a.out_csv, index=False, encoding='utf-8-sig')
    plt.figure(figsize=(4,4))
    if len(bin_df):
        plt.plot(bin_df['avg_conf'], bin_df['accuracy'], marker='o', label='observed')
    plt.plot([0,1], [0,1], linestyle='--', label='perfect')
    plt.xlabel('Average confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability diagram')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(a.out_png, dpi=180)
    print('saved:', a.out_csv, a.out_png)
    print(out.head(30).to_string(index=False))

if __name__ == '__main__':
    main()

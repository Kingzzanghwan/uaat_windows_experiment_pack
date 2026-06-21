import argparse, os, shutil, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='per-sample CSV with path, score, error, auto columns')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)
    required = {'error','score'}
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f'Missing columns: {miss}')
    if 'auto' in df.columns:
        bad = df[(df['auto'] == 1) & (df['error'] == 1)].copy()
    else:
        print('WARNING: no auto column. Using error=1 and high score as high-risk failure candidates.')
        bad = df[df['error'] == 1].copy()
    bad = bad.sort_values('score', ascending=False).head(a.k)
    bad.to_csv(os.path.join(a.out_dir, 'failure_candidates.csv'), index=False, encoding='utf-8-sig')
    path_col = None
    for cand in ['path','file','filename','image_path']:
        if cand in df.columns:
            path_col = cand; break
    copied = 0
    if path_col:
        for i, (_, r) in enumerate(bad.iterrows()):
            src = str(r[path_col])
            if os.path.exists(src):
                ext = os.path.splitext(src)[1] or '.png'
                dst = os.path.join(a.out_dir, f'fail_{i:02d}_score_{float(r["score"]):.3f}{ext}')
                shutil.copy(src, dst); copied += 1
    print('saved candidates CSV and copied images:', copied)
    print('out_dir:', a.out_dir)

if __name__ == '__main__':
    main()

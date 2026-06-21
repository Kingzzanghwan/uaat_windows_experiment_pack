import argparse, glob, os, re, pandas as pd

def parse_meta(path):
    name = os.path.basename(path.rstrip('\\/')).lower()
    cov = None; cost = None
    m = re.search(r'cov(?:erage)?[_-]?(\d+)', name)
    if m:
        cov = int(m.group(1)) / 100.0
    m = re.search(r'(?:c|cw|cost)[_-]?(\d+)', name)
    if m:
        cost = int(m.group(1))
    return cov, cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pattern', required=True, help='example: cost_c*_cov*')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.root, a.pattern)))
    rows = []
    for d in dirs:
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        cov, cost = parse_meta(d)
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rr = r.to_dict()
            rr['source_dir'] = d
            rr['target_coverage'] = cov
            rr['c_wrong'] = cost
            rows.append(rr)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print('saved:', a.out)
    print(out.head(40).to_string(index=False))

if __name__ == '__main__':
    main()

import argparse, os, re, pandas as pd

def parse_target_from_name(path):
    name = os.path.basename(path.rstrip('\\/')).lower()
    m = re.search(r'cov(?:erage)?[_-]?(\d+)', name)
    if m:
        v = int(m.group(1))
        return v / 100.0 if v > 1 else float(v)
    m = re.search(r'target[_-]?(0\.\d+|1\.0)', name)
    if m:
        return float(m.group(1))
    return None

def parse_cost_from_name(path):
    name = os.path.basename(path.rstrip('\\/')).lower()
    m = re.search(r'(?:c|cw|cost)[_-]?(\d+)', name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True, help='policy folders containing metrics.csv')
    ap.add_argument('--targets', nargs='*', type=float, default=None, help='optional target coverages matching roots order')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for i, root in enumerate(a.roots):
        f = os.path.join(root, 'metrics.csv')
        if not os.path.exists(f):
            print('skip missing:', f)
            continue
        df = pd.read_csv(f)
        target = a.targets[i] if a.targets and i < len(a.targets) else parse_target_from_name(root)
        cost = parse_cost_from_name(root)
        for _, r in df.iterrows():
            d = r.to_dict()
            d['source_dir'] = root
            d['target_coverage'] = target
            d['c_wrong_from_dir'] = cost
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print('saved:', a.out)
    print(out.head(30).to_string(index=False))

if __name__ == '__main__':
    main()

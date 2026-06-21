import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--base', default='fixed')
    ap.add_argument('--prop', default='uaat_monotone')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    need = {'policy','coverage','risk'}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f'Missing columns: {miss}')
    base = df[df['policy'] == a.base].copy()
    prop = df[df['policy'] == a.prop].copy()
    rows = []
    for _, pr in prop.iterrows():
        b = base.iloc[(base['coverage'] - pr['coverage']).abs().argsort().iloc[0]]
        rows.append({
            'target_coverage': pr.get('target_coverage', None),
            'policy_base': a.base,
            'policy_prop': a.prop,
            'coverage_base': b['coverage'],
            'coverage_prop': pr['coverage'],
            'coverage_gap': abs(float(b['coverage']) - float(pr['coverage'])),
            'risk_base': b['risk'],
            'risk_prop': pr['risk'],
            'risk_delta_base_minus_prop': float(b['risk']) - float(pr['risk']),
            'wrong_auto_base': b.get('wrong_auto_rate', None),
            'wrong_auto_prop': pr.get('wrong_auto_rate', None),
            'source_base': b.get('source_dir', None),
            'source_prop': pr.get('source_dir', None),
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print('saved:', a.out)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()

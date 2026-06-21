import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--group_col', default=None)
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        raise SystemExit(f'NOT FOUND: {a.csv}')

    df = pd.read_csv(a.csv)
    print('file:', a.csv)
    print('n rows:', len(df))
    print('columns:', ', '.join(df.columns))

    required = ['score', 'error']
    for c in required:
        print(f'{c}:', 'OK' if c in df.columns else 'MISSING')

    ctx = [c for c in df.columns if c.startswith('ctx_')]
    print('ctx_* columns:', len(ctx), ctx[:10])

    if 'uncertainty' in df.columns:
        print('uncertainty: OK')
    else:
        print('uncertainty: MISSING')

    if 'split' in df.columns:
        print('\nsplit counts')
        print(df['split'].value_counts(dropna=False).to_string())

    if a.group_col and a.group_col in df.columns:
        print(f'\n{a.group_col} counts')
        print(df[a.group_col].value_counts(dropna=False).head(30).to_string())

    if 'error' in df.columns:
        print('\nerror mean:', float(df['error'].mean()))

if __name__ == '__main__':
    main()

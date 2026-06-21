import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--id_col', default=None, help='path, sample_id, or image_id if available')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rows = []
    if 'split' not in df.columns:
        rows.append({'check': 'split_column', 'status': 'MISSING', 'detail': 'features.csv has no split column. Explain split in Methods or add split column during extraction.'})
    else:
        counts = df['split'].value_counts(dropna=False).to_dict()
        rows.append({'check': 'split_counts', 'status': 'OK', 'detail': str(counts)})
    id_col = a.id_col
    if id_col is None:
        for cand in ['path','sample_id','image_id','filename','file']:
            if cand in df.columns:
                id_col = cand; break
    if id_col and id_col in df.columns and 'split' in df.columns:
        overlaps = []
        splits = list(df['split'].dropna().unique())
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                a_ids = set(df[df['split']==splits[i]][id_col].astype(str))
                b_ids = set(df[df['split']==splits[j]][id_col].astype(str))
                overlaps.append((splits[i], splits[j], len(a_ids & b_ids)))
        rows.append({'check': 'id_overlap_between_splits', 'status': 'OK' if all(x[2]==0 for x in overlaps) else 'WARNING', 'detail': str(overlaps)})
    else:
        rows.append({'check': 'id_overlap_between_splits', 'status': 'SKIPPED', 'detail': 'No id/path column or no split column found.'})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    print(out.to_string(index=False))
    print('saved:', a.out)

if __name__ == '__main__':
    main()

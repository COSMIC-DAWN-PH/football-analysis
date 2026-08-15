import json, os
root = r'C:\Personal Profile\Profile\Football-Analysis-field-detection\eval\ball_crops'
for src in ['demo4','demo1']:
    sheets_path = os.path.join(root, src, 'sheets', 'sheets.jsonl')
    out_path = os.path.join(root, src, 'sheet_ids.json')
    mapping = []
    with open(sheets_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            mapping.append({'sheet': obj['sheet'], 'ids': obj['ids']})
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f)
    print(f'{src}: {len(mapping)} sheets, {sum(len(s["ids"]) for s in mapping)} ids')

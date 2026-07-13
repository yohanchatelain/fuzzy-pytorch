import os

def parse_logs(log_dir):
    data = {'attention': {}, 'mlp': {}, 'lm_head': {}}
    if not os.path.exists(log_dir):
        return data
        
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("Result ->"):
                    # Example line: Result -> Group: lm_head | Precision:  4 | Perplexity: 14684.98 | Time: 390.28s
                    parts = line.split('|')
                    group = parts[0].split(':')[1].strip()
                    prec = int(parts[1].split(':')[1].strip())
                    ppl = float(parts[2].split(':')[1].strip())
                    
                    if group in data:
                        data[group][prec] = ppl
    return data

contexts = ['128', '256', '384']
groups = ['attention', 'mlp', 'lm_head']

print("| Context | Component | Precision (bits) | SR Perplexity | RN Perplexity |")
print("|---|---|---|---|---|")

for ctx in contexts:
    sr_dir = os.path.join('perplexity_logs', ctx, 'ppl_percomponents_sr')
    rn_dir = os.path.join('perplexity_logs', ctx, 'ppl_percomponents_rn')
    
    sr_data = parse_logs(sr_dir)
    rn_data = parse_logs(rn_dir)
    
    for group in groups:
        # Iterate over all precisions (4,6,8,10) to produce a complete table
        for p in [4, 6, 8, 10]:
            sr_val = sr_data[group].get(p, 'N/A')
            rn_val = rn_data[group].get(p, 'N/A')
            if isinstance(sr_val, float):
                sr_val = f"{sr_val:.2f}"
            if isinstance(rn_val, float):
                rn_val = f"{rn_val:.2f}"
            print(f"| {ctx} | {group} | {p} | {sr_val} | {rn_val} |")

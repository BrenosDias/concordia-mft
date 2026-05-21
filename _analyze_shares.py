import json
import numpy as np
from pathlib import Path

with open("mfq_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)
with open("results.json", encoding="utf-8") as f:
    results = json.load(f)


def compute_mfq_scores(answers, mapping):
    scores = {}
    for foundation, questions in mapping.items():
        values = [answers[q] for q in questions if q in answers]
        scores[foundation] = float(np.mean(values)) if values else float("nan")
    return scores


shares = {
    "Maria Aparecida Silva": 11,
    "Patrícia Vasconcelos": 9,
    "Gabriel Nascimento": 8,
    "Josefa S. S. Dias": 8,
    "Leandro Pereira": 8,
    "Renilton Dias": 8,
    "Ana Paula Santos": 7,
    "Patrícia Almeida": 7,
    "Ricardo Menezes": 7,
    "José Antônio Ribeiro": 7,
    "Diego Santos": 7,
    "Andreia Santos": 7,
    "Bruna Carvalho": 6,
    "Paulo Ricardo Mendes": 6,
    "Brenda Souza Dias": 6,
    "Leandro Silva": 6,
    "Camila Barreto": 6,
    "Fernanda Oliveira": 6,
    "Juliana Carvalho": 6,
    "Rosângela Oliveira": 5,
}
actions = {
    "Maria Aparecida Silva": 83,
    "Patrícia Vasconcelos": 62,
    "Gabriel Nascimento": 101,
    "Josefa S. S. Dias": 44,
    "Leandro Pereira": 53,
    "Renilton Dias": 44,
    "Ana Paula Santos": 63,
    "Patrícia Almeida": 82,
    "Ricardo Menezes": 81,
    "José Antônio Ribeiro": 56,
    "Diego Santos": 64,
    "Andreia Santos": 66,
    "Bruna Carvalho": 55,
    "Paulo Ricardo Mendes": 86,
    "Brenda Souza Dias": 61,
    "Leandro Silva": 51,
    "Camila Barreto": 94,
    "Fernanda Oliveira": 60,
    "Juliana Carvalho": 67,
    "Rosângela Oliveira": 61,
}

by_name = {r["persona"]: compute_mfq_scores(r["answers"], mapping) for r in results}
foundations = ["care", "fairness", "loyalty", "authority", "purity"]

print("=== TOP 15: MFQ + shares ===")
top15 = sorted(shares.items(), key=lambda x: -x[1])[:15]
for name, sh in top15:
    s = by_name[name]
    vec = [s[k] for k in foundations]
    rate = sh / actions[name]
    print(
        f"{name:30} sh={sh:2d} rate={rate:.3f} "
        f"care={vec[0]:.2f} fair={vec[1]:.2f} loy={vec[2]:.2f} "
        f"auth={vec[3]:.2f} pur={vec[4]:.2f} mean={np.mean(vec):.2f}"
    )

# All 40 personas - need share data from logs
agg_dirs = list(Path(".").glob("**/correlations.csv"))
print("\nagg dirs with correlations:", agg_dirs)

# Compare high share-rate vs volume
print("\n=== High share RATE (>=0.15) among listed ===")
for name, sh in sorted(shares.items(), key=lambda x: -(x[1] / actions[x[0]])):
    rate = sh / actions[name]
    if rate >= 0.15:
        s = by_name[name]
        print(f"  {name}: rate={rate:.3f} shares={sh} dominant={max(s, key=s.get)}")

# Full population MFQ stats
all_mfq = []
for r in results:
    s = by_name[r["persona"]]
    all_mfq.append([s[k] for k in foundations])

all_mfq = np.array(all_mfq)
print("\n=== Population MFQ means (all 40) ===")
for i, f in enumerate(foundations):
    print(f"  {f}: mean={all_mfq[:, i].mean():.2f} std={all_mfq[:, i].std():.2f}")

top15_mfq = np.array([[by_name[n][f] for f in foundations] for n, _ in top15])
print("\n=== Top15 sharers MFQ means ===")
for i, f in enumerate(foundations):
    print(f"  {f}: {top15_mfq[:, i].mean():.2f} (pop: {all_mfq[:, i].mean():.2f})")

# Binding vs individualizing (Graham Haidt)
binding = ["loyalty", "authority", "purity"]
individualizing = ["care", "fairness"]
for label, cols in [("binding", binding), ("individualizing", individualizing)]:
    top_vals = [np.mean([by_name[n][c] for c in cols]) for n, _ in top15]
    pop_vals = [np.mean([by_name[r["persona"]][c] for c in cols]) for r in results]
    print(f"\n{label}: top15 mean={np.mean(top_vals):.2f} population={np.mean(pop_vals):.2f}")

# Low sharers in full 40 - grep results for personas with 0-2 shares if we find log
for path in Path(".").rglob("*.jsonl"):
    print("found jsonl:", path)

for path in Path(".").rglob("correlations.csv"):
    print("correlations:", path.read_text()[:500])

# Bottom sharers among all 40 (need shares for all - approximate from user not having full table)
# Compute all MFQ and list personas sorted by hypothetical - use results only for MFQ extremes
print("\n=== Lowest MFQ personas (potential low sharers) ===")
rows = []
for r in results:
    s = by_name[r["persona"]]
    vec = [s[k] for k in foundations]
    rows.append((r["persona"], np.mean(vec), s))
rows.sort(key=lambda x: x[1])
for name, mean, s in rows[:8]:
    sh = shares.get(name, "?")
    print(f"  {name:30} mean_mfq={mean:.2f} shares={sh} loy={s['loyalty']:.2f} auth={s['authority']:.2f}")

print("\n=== Highest MFQ personas ===")
for name, mean, s in rows[-8:]:
    sh = shares.get(name, "?")
    print(f"  {name:30} mean_mfq={mean:.2f} shares={sh}")

# Topic resonance proxy: cosine with care-heavy hospital news
hospital = {"care": 5.0, "fairness": 4.0, "loyalty": 3.0, "authority": 2.0, "purity": 2.0}
religious = {"care": 2.0, "fairness": 3.0, "loyalty": 4.0, "authority": 5.0, "purity": 5.0}

def cos(a, b):
    va = np.array([a[k] for k in foundations])
    vb = np.array([b[k] for k in foundations])
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

print("\n=== Resonance with hospital_donation (care-heavy) for top sharers ===")
for name, sh in top15:
    s = by_name[name]
    print(f"  {name[:28]:28} hospital_cos={cos(s, hospital):.3f} religious_cos={cos(s, religious):.3f}")

# P0 falsification (Naive-RAG contexts)

n=700 budget=600. Logit: hallu ~ lam2 + mean_rel + tokens (z-scored).

- bridge lam2: coef=-0.145 p=0.2357 (hallu rate 0.316, mean lam2 0.184, n=500)
- bridge lam2_raw: coef=-0.075 p=0.5422 (hallu rate 0.316, mean lam2_raw 0.137, n=500)
- comparison lam2: coef=-0.458 p=0.2005 (hallu rate 0.175, mean lam2 0.062, n=200)
- comparison lam2_raw: coef=-0.767 p=0.2860 (hallu rate 0.175, mean lam2_raw 0.040, n=200)
- all lam2: coef=-0.104 p=0.3202 (hallu rate 0.276, mean lam2 0.149, n=700)
- all lam2_raw: coef=-0.057 p=0.5870 (hallu rate 0.276, mean lam2_raw 0.109, n=700)

Hallucination rate by connectivity (bridge): ncomp=1: 0.263 (n=114), ncomp=2: 0.321 (n=137), ncomp=3: 0.337 (n=249)
# P1 diagnostic: AUROC for detecting hallucination (0.5 = chance)

Score = -lam2 (low connectivity => predicted hallucination); controls: -mean_rel, tokens.

| scope | system | n | hallu rate | AUROC(-lam2) | AUROC(-lam2_raw) | AUROC(-mean_rel) | AUROC(-ncomp==1) |
|---|---|---|---|---|---|---|---|
| bridge | pooled | 3000 | 0.331 | 0.520 | 0.524 | 0.514 | 0.513 |
| bridge | graphrag | 500 | 0.332 | 0.500 | 0.527 | 0.509 | 0.523 |
| bridge | gretriever | 500 | 0.342 | 0.515 | 0.509 | 0.530 | 0.494 |
| bridge | hybrid | 500 | 0.350 | 0.516 | 0.528 | 0.526 | 0.516 |
| bridge | naive | 500 | 0.316 | 0.537 | 0.532 | 0.550 | 0.528 |
| bridge | ses | 500 | 0.296 | 0.515 | 0.514 | 0.490 | 0.510 |
| bridge | spectrag | 500 | 0.350 | 0.511 | 0.509 | 0.496 | 0.488 |
| comparison | pooled | 1200 | 0.188 | 0.517 | 0.552 | 0.518 | 0.538 |
| comparison | graphrag | 200 | 0.225 | 0.562 | 0.578 | 0.541 | 0.537 |
| comparison | gretriever | 200 | 0.205 | 0.497 | 0.528 | 0.530 | 0.527 |
| comparison | hybrid | 200 | 0.190 | 0.453 | 0.493 | 0.443 | 0.517 |
| comparison | naive | 200 | 0.175 | 0.478 | 0.543 | 0.483 | 0.540 |
| comparison | ses | 200 | 0.145 | 0.589 | 0.611 | 0.588 | 0.558 |
| comparison | spectrag | 200 | 0.185 | 0.491 | 0.552 | 0.526 | 0.524 |
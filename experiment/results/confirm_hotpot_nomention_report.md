# Confirmatory results: hotpot_nomention (bridge only, n=500, budget 600, frozen params)

| system     |   full |   sp_recall |   lam2 |   tokens |      f1 |      em |   hallu |   ungrounded |   abstain |   acc_answered |
|:-----------|-------:|------------:|-------:|---------:|--------:|--------:|--------:|-------------:|----------:|---------------:|
| graphrag   |  0.582 |       0.772 |  0.209 |  573.394 | nan     | nan     | nan     |      nan     |   nan     |        nan     |
| gretriever |  0.576 |       0.768 |  0.192 |  582.054 | nan     | nan     | nan     |      nan     |   nan     |        nan     |
| hybrid     |  0.518 |       0.74  |  0.126 |  587.898 | nan     | nan     | nan     |      nan     |   nan     |        nan     |
| naive      |  0.52  |       0.737 |  0.166 |  582.64  | nan     | nan     | nan     |      nan     |   nan     |        nan     |
| pool_rel   |  0.574 |       0.77  |  0.189 |  582.132 |   0.422 |   0.284 |   0.41  |        0.066 |     0.14  |          0.523 |
| ses        |  0.564 |       0.76  |  0.299 |  582.516 |   0.405 |   0.276 |   0.396 |        0.064 |     0.172 |          0.522 |
| spectrag   |  0.574 |       0.77  |  0.189 |  582.132 |   0.422 |   0.284 |   0.41  |        0.066 |     0.14  |          0.523 |

## SES-RAG minus baseline (paired bootstrap, one-sided; Holm over all rows below)

| baseline | metric | diff | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| naive | full | +0.044 | [+0.006, +0.084] | 0.0134 | 0.0938 |
| hybrid | full | +0.046 | [+0.010, +0.084] | 0.0080 | 0.0640 |
| pool_rel | full | -0.010 | [-0.042, +0.024] | 0.7498 | 1.0000 |
| pool_rel | f1 | -0.017 | [-0.044, +0.010] | 0.8918 | 1.0000 |
| graphrag | full | -0.018 | [-0.056, +0.020] | 0.8434 | 1.0000 |
| spectrag | full | -0.010 | [-0.042, +0.024] | 0.7498 | 1.0000 |
| spectrag | f1 | -0.017 | [-0.044, +0.010] | 0.8918 | 1.0000 |
| gretriever | full | -0.012 | [-0.044, +0.020] | 0.7884 | 1.0000 |

**Frozen pass criterion (beat every baseline incl. mu=0 control on complete-retrieval and F1, Holm p<0.05): NOT PASSED**

## Bridge-failure link: outcome by whether the context contains both gold paragraphs

|   full |   ('correct', 'mean') |   ('hallu', 'mean') |   ('ungrounded', 'mean') |   ('abstain', 'mean') |   ('f1', 'mean') |
|-------:|----------------------:|--------------------:|-------------------------:|----------------------:|-----------------:|
|      0 |                 0.189 |               0.508 |                    0.059 |                 0.303 |            0.181 |
|      1 |                 0.636 |               0.328 |                    0.07  |                 0.036 |            0.593 |

n per group: {0.0: 644, 1.0: 856}

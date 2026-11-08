# SES-RAG bridge-only experiment: final summary (v2)

Protocol (SPEC.md v2): tuning on dev + seen questions; 500 untouched HotpotQA bridge holdout questions; frozen settings; context order shuffled; Holm correction; pass criterion fixed in advance. Second dataset: MuSiQue (settings transferred, not re-tuned). Budget 600 tokens, gemma3, OpenAI embeddings.

## Results
| Test | SES-RAG vs the field | Frozen criterion |
|---|---|---|
| HotpotQA bridge holdout (n=500) | Best of 7 on gold-complete retrieval (0.624) and F1 (0.441). Holm-significant vs naive, hybrid (both metrics), vs mu=0 control and G-Retriever-lite (retrieval only). Not significant vs SpectRAG-lite, GraphRAG-lite | NOT PASSED |
| Same, graph without title-mention edges | No advantage over any graph baseline (0.564 vs 0.574-0.582) | NOT PASSED |
| MuSiQue (n=500, transfer) | Ties naive/control/SpectRAG/GraphRAG/G-Retriever; beats hybrid only | NOT PASSED |

## What is supported
- Bridge failure is real and costly: with the bridge paragraph present, correct answers are 64% (HotpotQA) vs 19% when it is missing; abstentions 4% vs 32%.
- On HotpotQA, a connectivity-seeking selector on a link-aware graph retrieves bridge-complete contexts more often than flat retrieval (+10 pts) and modestly more than other graph methods (+2 to +6 pts, not significant).

## What is not supported
- lambda2 as the mechanism (comp/density/raw variants tie), lambda2 as a hallucination predictor (P1 AUROC ~0.52), generalisation to MuSiQue, and any advantage without the title-mention edges.

## Caveats
Baselines are simplified re-implementations on a shared graph; hallucination = answered and wrong (abstention can lower it); single generator run; 2Wiki, LightRAG, HippoRAG 2 not run; title-mention edges mirror HotpotQA's hyperlink construction.

## Options
1. Negative/diagnostic paper: bridge failure quantified + honest evaluation of connectivity-aware selection (works only with link-aware graphs).
2. Improve the method for harder chains (MuSiQue retrieves both gold only ~20%): e.g. iterative expansion from selected nodes, entity-linking edges from an LLM, then re-test on a fresh holdout (361 HotpotQA bridge questions remain unused).
3. Pre-register lambda2-raw (+3.4 pts retrieval on the HotpotQA holdout, seen) as the objective and confirm on the unused questions.

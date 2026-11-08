# SES-RAG experiment summary (2026-11-06/07)

Setup: HotpotQA distractor, 2,000-question sample (19,264 paragraphs pooled), OpenAI text-embedding-3-small, gemma3 via Ollama, 600-token budget, eval = 500 bridge + 200 comparison (disjoint from the 300-question dev split used for tuning). Baselines are simplified re-implementations on one shared graph (D-017).

## Findings
1. **Bridge questions: SES-RAG beats Naive, Hybrid, SpectRAG-lite, GraphRAG-lite, G-Retriever-lite** on F1 (+4 to +7 pts), gold-complete retrieval (+7 to +14 pts), paired-bootstrap CIs exclude 0 (p2_report.md). Holds at budgets 300-1500 on the bridge-only budget sweep (p3_report.md).
2. **Comparison questions: SES-RAG loses** (F1 0.28 vs ~0.40; gold-complete retrieval 0.59 vs ~0.9). On the mixed 700, SES does not beat SpectRAG-lite/GraphRAG-lite/G-Retriever-lite on F1.
3. **Hallucination "win" is mostly retrieval recall plus abstention.** Given a gold-complete context all systems hallucinate ~25-28% (audit).
4. **The theory is not supported:** P0 lambda2 coefficient p=0.24 (spec bar p<0.05 not met); P1 AUROC ~0.52 (chance).
5. **lambda2 is not shown to be the mechanism:** mu=0 beats full SES on the mixed set; density/component-count/raw-lambda2 variants tie on bridge.

## Honest claim the data supports
"A graph-expanded, connectivity-seeking selector improves multi-hop (bridge) retrieval and answer F1 over flat and graph baselines; lambda2 specifically, and the lambda2-hallucination hypothesis, are not supported at this scale."

## Follow-ups (cheapest first, from the audit)
- Report hallucination conditional on full support; abstention-adjusted accuracy
- Bootstrap CIs for the P3 ablations, bridge only; Holm correction
- Rebuild graph with title-mention weight 0 and rerun retrieval-only comparison
- Give every baseline the same pool/relevance and the same tuning grid
- Type-gated mu (mu=0 on comparison questions), tuned on dev only, labelled post-hoc
- Add an LLM judge / shuffled context order; 2WikiMultiHopQA; originals of LightRAG/HippoRAG2

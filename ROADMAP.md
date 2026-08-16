# RAGinGoa — 5-Day Build Roadmap (HH Goa 2026 Task 2)

## 1. Status header
- **Task 1 (Frame247):** Shipped, live at [frame247.vercel.app](https://frame247.vercel.app)
- **Task 2 (RAGinGoa):** 5-day compressed build starting Aug 17, 2026
- **Deadline:** Aug 22, 2026, 11:59 PM — **No resubmissions allowed**

> [!IMPORTANT]
> **Aug 22 is buffer only, not a build day** — Freeze code and submit all deliverables by the end of Aug 21.

---

## 2. Role split (by concern, not frontend/backend)

### Person A — "Get the right text" (Data, Retrieval, Chunking)
- [ ] MSMARCO-XI loading
- [ ] All three chunking strategies (fixed-size+overlap, semantic, metadata/structure-aware — kept simple: passage-boundary chunking + doc id/language/position metadata, not a from-scratch structure parser)
- [ ] Qdrant indexing (3 collections)
- [ ] The `retrieve(query, k) -> list[Chunk]` function
- [ ] Retrieval evaluation (recall@k / MRR on a 20-30 query held-out set), chunking comparison writeup

### Person B — "Do something with the right text" (Voice, Generation, Guardrails, Harness, Frontend)
- [ ] Sarvam STT integration
- [ ] Guardrail logic (off-topic, unsafe input, grounding check) — hallucination check as a cheap heuristic (entities/numbers not in retrieved chunks), skip a second-LLM verifier call unless time allows
- [ ] Generation wiring (Gemini API, forced citation format)
- [ ] FastAPI harness stringing all stages together (logging/timing/retries)
- [ ] Latency benchmark script (P50/P70/P100 on 30-50 queries)
- [ ] Frontend (mic button, waveform, answer + citations, latency readout — minimal, no extra polish) + deploy

---

## 3. Shared contract (agree Day 1, before writing code)

```python
class Chunk:
    text: str
    score: float
    source_id: str
    strategy: str   # "fixed" | "semantic" | "structured"
    metadata: dict

def retrieve(query: str, k: int = 5) -> list[Chunk]:
    ...
```

> [!NOTE]
> Person B builds against a stub `retrieve()` (hardcoded fake Chunks) from Day 1. Real swap-in happens Day 2 evening — this is the critical pinch point in the compressed schedule (see Day 2 note below).

---

## 4. Day-by-day schedule

| Day | Date | Person A | Person B | Together? |
|---|---|---|---|---|
| 1 | Aug 17 | [ ] Agree contract.<br>[ ] MSMARCO-XI loaded.<br>[ ] Fixed-size chunking done + indexed.<br>[ ] Start semantic chunking. | [ ] Agree contract.<br>[ ] Sarvam + Gemini API keys live.<br>[ ] FastAPI skeleton up.<br>[ ] Stub `retrieve()` in place.<br>[ ] STT wired end-to-end against stub. | Morning: contract + kickoff call |
| 2 | Aug 18 | [ ] Finish semantic chunking.<br>[ ] Metadata/structure-aware chunking (kept simple).<br>[ ] All 3 collections indexed by EOD. | [ ] Guardrails (off-topic + unsafe + grounding threshold) built against stub.<br>[ ] Generation wiring against stub. | Evening: swap stub → real `retrieve()`, integration smoke test |
| 3 | Aug 19 | [ ] Run recall@k / MRR across all 3 strategies.<br>[ ] Pick winner (or hybrid).<br>[ ] Write chunking comparison table into README. | [ ] Latency benchmark script (`bench.py`) run against live chunking.<br>[ ] Fix integration bugs from Day 2 night. | Debug integration together as issues surface |
| 4 | Aug 20 | [ ] Help debug.<br>[ ] Review generation quality against real chunks.<br>[ ] Finalize chunking writeup. | [ ] Frontend (mic, waveform, answer+citations, latency readout).<br>[ ] Deploy backend + frontend.<br>[ ] Test live link fresh in incognito. | Full day together: integration hardening + deploy |
| 5 | Aug 21 | [ ] Write README (architecture diagram, chunking table, latency numbers).<br>[ ] Cross-brief B on retrieval half. | [ ] Cross-brief A on pipeline/frontend half.<br>[ ] Final live-link check. | Record both videos together, post individually to IG/X/LinkedIn with #RAGInGoa, submit form |
| Buffer | Aug 22 | [ ] Fix anything broken.<br>[ ] Recheck live link.<br>[ ] Confirm submission went through.<br>[ ] No new features. | [ ] Same. | As needed |

---

## 5. Compression risk callout

> [!WARNING]
> **Pinch Point:** Integration (Day 2 evening) is the critical pinch point — much tighter than a normal schedule would allow. If fixed+semantic chunking isn't done by Day 1 EOD, that's the trigger to simplify metadata chunking further or drop to a 2-way strategy comparison rather than lose Day 3. Flag this daily during check-ins.

---

## 6. Working logistics

- **Comms:** Async check-in every morning (5-10 min), one call per "together" block (Day 1 AM, Day 2 PM, Day 3, Day 4, Day 5)
- **Code:** Shared repo, feature branches (`feat/chunking-*`, `feat/pipeline-*`), merge to main daily — no end-of-project big merge
- **Secrets:** `.env.example` with variable names only, real keys shared via private channel — never in group chat
- **Pairing:** Screen-share Day 2 integration and Day 4 deploy, swap driver every 20-30 min

---

## 7. Open items (still unresolved)

> [!WARNING]
> **Unresolved Items:**
> - [ ] Confirm: individual vs. shared submission requirement (check task page / ask organizers) — do this Day 1, it affects deploy planning
> - [ ] Who owns the production Qdrant instance / API keys
> - [ ] Confirm both people's availability holds for all 5 days — this schedule has near-zero slack

---

## 8. Submission checklist (from PRD)

- [ ] GitHub repo, public, README with architecture diagram + chunking comparison table
- [ ] Live working link (mic works over HTTPS)
- [ ] P50/P70/P100 latency numbers in README, from real benchmark run
- [ ] Guardrails demonstrated in demo video (off-topic + unanswerable query)
- [ ] Video 1 (90s): team/process, not the product
- [ ] Video 2: end-to-end demo including a correct refusal case
- [ ] Both videos posted to Instagram, X, LinkedIn by both members, #RAGInGoa on every post
- [ ] At least one team member's Instagram is public
- [ ] Form submitted (per person if individual submission required)

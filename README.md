# CONSEQUENCE

> **Before you change one thing, see what else breaks.**

CONSEQUENCE is a counterfactual operations intelligence product. It turns operational evidence into a structured dependency graph, applies a proposed change, deterministically propagates downstream effects, ranks consequences, and produces an executive decision recommendation with required actions.

## Product flow

```text
Evidence
   ↓
Entities + relationships
   ↓
Proposed change / mutation
   ↓
Deterministic propagation
   ↓
Impact scoring
   ↓
Business consequence interpretation
   ↓
Executive decision
   ↓
Customer report
```

## Customer experience

- `sales.html` — customer-facing product page
- `commercial-mvp.html` — premium interactive demo
- `pilot.html` — customer pilot: paste evidence + proposed change → assessment
- `server.py` — API + deterministic fallback engine
- `/api/health` — deployment health check
- `/api/provider/status` — provider status

## Run locally

```bash
python server.py
```

Open `http://127.0.0.1:4173/`.

No third-party Python package is required for the current server. When no AI provider is configured, the product uses the deterministic fallback assessment so the workflow remains demonstrable.

## Public deployment

The repository includes `render.yaml` for a Render Web Service. The server reads Render's `PORT` environment variable and binds to `0.0.0.0` when running on Render.

For live AI, configure the provider credentials as server-side environment variables. Never commit API keys to the repository.

## Commercial positioning

**Start with one real operational decision.**

Customer workflow:

1. Customer provides evidence they are authorized to share.
2. Customer describes one proposed operational change.
3. CONSEQUENCE produces an inspectable impact assessment.
4. The team validates the highest-priority dependencies.
5. The result becomes a customer-specific pilot and repeatable workflow.

The current implementation includes healthcare, education, business, and infrastructure demonstration scenarios. Customer-specific rules should be validated with the responsible domain owner before production decisions.

## Safety / decision support

CONSEQUENCE is decision-support software. Its outputs are not a substitute for the responsible medical, legal, safety, operational, or technical owner's judgment.

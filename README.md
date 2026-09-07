# CONSEQUENCE

CONSEQUENCE is a deterministic counterfactual clinic simulator with an evidence-grounded AI reasoning boundary.

## Run

Fixture/demo mode needs no key:

```bash
python3 server.py
```

Open `http://127.0.0.1:4173/consequence-demo.html`.

Live OpenAI mode reads the key only on the server process:

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='gpt-5.6-luna'
python3 server.py
```

The browser shows `LIVE PROVIDER` only when the server reports a configured key. If the key is missing or a provider request fails, no AI output is accepted and the app remains honest about fixture/unavailable mode.

Inside a Browser Use V4 runtime, the server automatically uses the runtime's secure OpenAI-compatible gateway when available; the gateway token never reaches the browser.

The current evidence uploader sends text content to the provider. PDF/image uploads remain metadata-only until a multimodal upload path is added.

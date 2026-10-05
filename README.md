# CONSEQUENCE



> Before you change one thing, see what else breaks.



CONSEQUENCE is an evidence-grounded AI system that builds a machine-readable model of a real-world system and simulates the cascading consequences of a proposed change before it happens.



It combines flexible AI reasoning with a structured, inspectable, deterministic simulation and scoring layer. The current implemented vertical is a clinic relocation demonstration.



## The Problem



A change that looks local can create consequences elsewhere because real-world systems contain hidden dependencies, constraints, capacity limits, accessibility requirements, schedules, and operational relationships.



Ordinary summaries, dashboards, and chatbots can describe what is already known, but they do not actually simulate the downstream effects of a proposed change through an interconnected system.



## What CONSEQUENCE Does



The system follows this pipeline:



```text

Evidence

   â†“

Structured Facts

   â†“

Entity / Relationship Graph

   â†“

Change Interpretation

   â†“

Counterfactual Simulation

   â†“

Consequence Reasoning

   â†“

Risk Scoring

   â†“

Impact Graph

   â†“

Mitigation Plan

## Commercial MVP

The repository now includes **CONSEQUENCE — AI Change Impact Simulator** in `commercial-mvp.html`.

The commercial MVP adds:
- Healthcare, Education, Business, and Infrastructure scenario presets
- Natural-language proposed-change workflow
- Current-world vs counterfactual comparison
- Prioritized consequence cards
- Impact-path visualization
- Explicit demo/live provider status
- Print-to-PDF report action
- A clear separation between confirmed dependencies and AI hypotheses

### Run locally

```bash
python server.py
```

Open `http://127.0.0.1:4173/`.

The root route opens the commercial MVP. The original `consequence-demo.html` remains available as the hackathon/demo experience.

> AI output is assistive and should be reviewed by qualified domain professionals before operational decisions.

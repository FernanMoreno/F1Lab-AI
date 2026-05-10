"""Optional LLM-assisted analysis layer for F1Lab-AI.

The LLM layer is strictly read-only relative to the simulation:
- it does NOT affect runtime simulation decisions
- it does NOT decide whether unsafe_legal_state is emitted
- it does NOT mutate metrics

It is only used for: summarizing audit reports, proposing synthetic family
candidates, and explaining counterfactual results in cautious language.

Requires the optional `nvidia` extra:
    pip install 'f1lab-ai[nvidia]'
"""

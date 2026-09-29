# H.27 experimental event-reader prompt addition (NOT applied to production)

This text was tested ONLY as an in-process monkeypatch of
`finance.documents.event_extractor._SYSTEM` during Phase H.27's isolated A/B
experiment. It was never written to `finance/documents/event_extractor.py` on
disk, and the production file is confirmed unmodified by this phase (empty
`git diff` on `finance/`, aside from pre-existing H.22 work).

Appended immediately before the production prompt's closing
`"Respond with ONLY a JSON object, no prose."` line:

```
16. EVENT TYPE ANSWERS "WHAT KIND OF INSTRUMENT OR TRANSACTION IS THIS", NOT
"WAS MONEY DRAWN": ISSUER_DEBT_ISSUANCE covers ANY new debt obligation the
company incurs or enters into -- registered notes or bonds, a private
placement of debt, a term loan, a revolving credit facility, or a bridge
facility -- whether or not anything has been drawn on it yet. A facility
being ESTABLISHED is still ISSUER_DEBT_ISSUANCE in event_type; funded=false,
committed=true is how you report that nothing has been drawn -- it is NEVER
a reason to pick a different event_type. Use REFINANCING only when a span
states the new debt specifically replaces or refinances EXISTING debt. Use
OTHER_MATERIAL_FINANCING ONLY when the transaction is a material financing
arrangement that is NOT itself a debt issuance, an equity issuance, or an
acquisition -- for example a lease, a guarantee, or a non-debt commercial
financing arrangement. A credit facility, term loan, or bridge facility is
NEVER OTHER_MATERIAL_FINANCING merely because it is undrawn or uncommitted --
that fact belongs in funded/committed, never in event_type. Worked example:
"the Company entered into a $500 million revolving credit facility; no
amounts were drawn at closing" -> event_type=ISSUER_DEBT_ISSUANCE,
funded=false, committed=true.
```

`sha256(revised _SYSTEM)` = `354cc7e1690b9ce943aed1f4ef4aa823203546f60709cd98c15b8c11e524dce9`
`sha256(production _SYSTEM)` = `2c3a6529c31abc313a9d7102edc2648989ab04b01af1e14b2b5808dd90852c27`

## Why this is not recommended for adoption yet

The measured effect (see `h27_variant_A_production_prompt.json` vs.
`h27_variant_B_revised_prompt.json`) is real and large on the dimension it
targets:

- `event_type_accuracy`: 0.52 -> 0.96 (all 3 runs, both variants)
- `event_accepted_recall`: 0.40/0.40/0.36 -> 0.72/0.64/0.68
- `item_code_consistency`: ~0.61 -> 0.96-1.0

But the mandated x3 controlled comparison also showed MORE hard-safety hits
under the revised prompt (3, across 3 distinct codes) than under the
production prompt (1) in the same run. Attribution tracing shows all 3 land
on pre-documented complex multi-amount/dual-currency fixtures unrelated to
rule 16's subject matter, and a separate 3-pass re-run of variant B alone
produced zero hits -- consistent with ordinary model-level stochastic
variability rather than a rule-16-caused regression, but the sample size is
too small to state that as a defensible conclusion rather than a plausible
one. Per phase H.27 spec section 9, this is scored INCONCLUSIVE on safety,
not READY. No further prompt-engineering iteration was attempted per
section 9's own instruction ("do not continue adding extraction heuristics
to compensate for model capability limitations").

If a future phase wants to pick this up: the recommended next step is a
larger, SAFETY-FOCUSED sample (10+ runs) of variant B specifically, not
another prompt revision -- to determine whether the observed hard-safety
rate under the revised prompt is actually distinguishable from the
production prompt's own non-zero baseline rate on this same model.

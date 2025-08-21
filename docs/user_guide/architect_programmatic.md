# Architect (Programmatic)

The CLI’s Architect is implemented programmatically in  using a
StateMachine step. This supersedes older YAML‑only flows while keeping a reference YAML example at
 for study and experimentation.

- Full state machine (GatheringContext → Planning → PlanApproval → ParameterCollection → Generation →
  Validation → DryRunOffer → Finalization)
- Validation/repair implemented via built‑ins (, )
- Interactive plan approval (HITL) supported via  and 

Enable the full conversational state machine in the CLI by setting:



For a deep dive on implementation and testing, see the in‑repo guide:
- GitHub: https://github.com/aandresalvarez/flujo/blob/main/flujo/architect/README.md

Related docs:
- User Guide: [Architect (YAML reference)](architect.md)
- Advanced: [Architect Flow Fundamentals](../architect_flow_first_principles.md)


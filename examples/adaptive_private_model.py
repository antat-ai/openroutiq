from pathlib import Path

from openroutiq import AdaptiveRouter


project = Path(__file__).resolve().parents[1]
router = AdaptiveRouter.from_file(
    project / "models.example.json",
    registry=project / ".openroutiq/adaptive-models.sqlite3",
)

router.encounter_opaque(
    model_id="customer/legal-v7:high",
    provider="customer-private",
    model="legal-v7",
    api_style="openai_compatible",
    base_url="https://models.customer.internal/v1",
    reasoning_level="high",
    max_context_tokens=64_000,
    capabilities=["text", "tools", "json_schema"],
    tasks=["legal_review"],
    latency_ms=800,
    input_price_per_million=0.40,
    output_price_per_million=0.80,
    local=True,
)

request = "Review this agreement against our internal legal policy."
decision = router.route(
    request,
    task="legal_review",
    pinned_model="customer/legal-v7:high",
)
print(decision.to_dict())

# Execute `decision` with the customer's existing framework/client, evaluate the complete
# task, and then provide the observed outcome. Never use the model's self-confidence here.
# router.record_evaluation(
#     request,
#     decision.selected.model_id,
#     task="legal_review",
#     quality_score=evaluate_complete_task(result),
#     latency_ms=measured_latency_ms,
#     actual_cost_usd=measured_cost_usd,
#     success=task_completed,
# )

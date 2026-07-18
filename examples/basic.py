from pathlib import Path

from openroutiq import Router

catalog = Path(__file__).resolve().parents[1] / "models.example.json"
decision = Router.from_file(catalog).route(
    [
        {"role": "system", "content": "You are reviewing production Python."},
        {"role": "user", "content": "Find and fix the race condition in this worker."},
    ],
    strategy="auto",
)

print(decision.selected.model_id)
print(decision.to_dict())

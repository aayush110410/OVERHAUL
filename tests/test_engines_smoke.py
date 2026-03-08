"""Quick smoke test for engine framework."""
import asyncio
import json

from engines import get_registry
from data_integration.bridge import (
    ncr_data_to_engine_input,
    build_scenario_from_prompt,
    format_engine_results_for_chat,
)

async def main():
    registry = get_registry()

    scenario = build_scenario_from_prompt(
        "What if Delhi adopts 20% EV fleet, expands metro by 50km, and adds congestion pricing?",
        city="delhi",
    )

    data = ncr_data_to_engine_input({}, city="delhi")
    results = await registry.run_scenario(scenario, data)

    print(f"Engines returned: {list(results.keys())}")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(f"  Confidence: {r.confidence}")
        m = json.dumps(r.metrics, indent=2, default=str)
        print(f"  Metrics: {m[:300]}")
        if r.recommendations:
            print(f"  Recs: {r.recommendations[:2]}")
        if r.warnings:
            print(f"  Warnings: {r.warnings}")

    formatted = format_engine_results_for_chat(results, scenario.name)
    print(f"\n=== Impact Cards ({len(formatted['impactCards'])}) ===")
    for card in formatted["impactCards"]:
        print(f"  [{card['direction']}] {card['metric']}: {card['value']} ({card.get('delta', '')})")

    print(f"\nRecommendations: {len(formatted['recommendations'])}")
    for rec in formatted["recommendations"][:5]:
        print(f"  - {rec}")

    print("\nSMOKE TEST PASSED")

if __name__ == "__main__":
    asyncio.run(main())

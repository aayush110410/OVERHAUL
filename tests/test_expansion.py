"""Quick smoke test for expanded architecture."""
import asyncio
import time

from engines import get_registry
from engines.base import Intervention, Scenario
from engines.scenarios import list_templates, build_scenario_from_template
from engines.geospatial import generate_geojson
from data_integration.bridge import (
    prompt_to_interventions,
    build_scenario_from_prompt,
    format_engine_results_for_chat,
)


def test_registry():
    reg = get_registry()
    print("=== Registered engines ===")
    for e in reg.list_engines():
        print(f"  {e['name']}: {e['capabilities']}")
    assert len(reg.list_engines()) >= 7, "Expected 7+ engines"
    return reg


def test_templates():
    print("\n=== Templates ===")
    templates = list_templates()
    for t in templates:
        print(f"  {t['id']}: {t['interventions']} interventions, domains={t['domains']}")
    assert len(templates) >= 6, "Expected 6+ templates"


def test_intervention_detection():
    print("\n=== Intervention detection ===")
    prompts = [
        "Add freight consolidation centers and micro-hubs for last mile delivery",
        "Implement work from home policy and odd even scheme with e-scooter sharing",
        "Build metro and BRT with EV charging",
    ]
    for p in prompts:
        intvs = prompt_to_interventions(p)
        names = [i.name for i in intvs]
        print(f"  \"{p[:60]}\" -> {names}")
        assert len(intvs) >= 2, f"Expected 2+ interventions for: {p}"


async def test_two_phase(reg):
    scenario = build_scenario_from_template("sustainable_ncr")
    data = {
        "baseline_pm25": 285,
        "baseline_speed_kmh": 25,
        "ev_count": 360000,
        "city": "delhi",
    }
    results = await reg.run_scenario(scenario, data)
    print("\n=== Two-phase execution ===")
    for name, r in results.items():
        domain = r.domain.value if hasattr(r.domain, "value") else r.domain
        key_metrics = list(r.metrics.keys())[:4] if r.metrics else []
        print(f"  {name} ({domain}): conf={r.confidence:.2f}, metrics={key_metrics}")
    assert len(results) >= 5, f"Expected 5+ engine results, got {len(results)}"

    # GeoJSON
    geo = generate_geojson(results)
    layers = sorted(set(f["properties"].get("layer", "?") for f in geo["features"]))
    print(f"\n=== GeoJSON: {len(geo['features'])} features ===")
    print(f"  Layers: {layers}")
    assert len(geo["features"]) >= 10, "Expected 10+ GeoJSON features"

    # Format for chat
    fmt = format_engine_results_for_chat(results, "test")
    print(f"\n=== Impact cards: {len(fmt['impactCards'])} ===")
    for c in fmt["impactCards"]:
        print(f"  {c['metric']}: {c['value']}")
    assert len(fmt["impactCards"]) >= 5, "Expected 5+ impact cards"

    # Cache test
    t0 = time.time()
    results2 = await reg.run_scenario(scenario, data)
    dt = time.time() - t0
    print(f"\n=== Cache hit: {dt*1000:.1f}ms ===")
    assert dt < 0.05, f"Cache miss — took {dt*1000:.0f}ms"


if __name__ == "__main__":
    reg = test_registry()
    test_templates()
    test_intervention_detection()
    asyncio.run(test_two_phase(reg))
    print("\n✅ All smoke tests passed")

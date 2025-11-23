from src.analysis.metrics import average_delay, mota, throughput, travel_time_stats


def test_travel_time_stats_basic():
    stats = travel_time_stats([10.0, 12.0, 8.0])
    assert stats["avg"] == 10.0
    assert stats["min"] == 8.0
    assert stats["max"] == 12.0


def test_throughput_converts_to_hour():
    result = throughput([10, 10], interval_seconds=60)
    assert result == 600  # 20 vehicles over 2 minutes -> 600 veh/hour


def test_average_delay_zero_when_below_free_flow():
    assert average_delay([9.0, 8.5], free_flow_time=10.0) == 0.0


def test_mota_handles_zero_denominator():
    assert mota(0, 0, 0, 0) == 0.0
    assert mota(10, 1, 2, 1) == 1 - (1 + 2 + 1) / 12

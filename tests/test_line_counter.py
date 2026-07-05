"""Unit tests for :class:`trafficsight.services.line_counter.VirtualLineCounter`."""

from __future__ import annotations

from trafficsight.domain.entities import LineCrossingEvent
from trafficsight.domain.ports import InMemoryDetectionRepository
from trafficsight.services.line_counter import VirtualLineCounter


def _make_counter(stream_url: str = "http://example/stream") -> VirtualLineCounter:
    return VirtualLineCounter(stream_url, repo=InMemoryDetectionRepository())


def test_no_crossing_when_far_away():
    counter = _make_counter()
    counter.update(1, 100, 100, "car", 30.0, "cam")
    result = counter.update(1, 100, 105, "car", 30.0, "cam")
    assert result is None
    assert counter.get_summary()["unique_total"] == 0


def test_horizontal_line_crossing_counts_once():
    counter = _make_counter()
    # Walk track 1 from y=200 down to y=400, crossing the "Utara" line at y=310.
    counter.update(1, 500, 200, "car", 30.0, "cam")
    result = counter.update(1, 500, 400, "car", 30.0, "cam")
    assert result is not None and "Utara" in result
    summary = counter.get_summary()
    assert summary["unique_total"] == 1
    assert summary["per_arm"]["Utara"]["masuk"] == 1


def test_same_track_does_not_recount_same_arm():
    counter = _make_counter()
    counter.update(1, 500, 200, "car", 30.0, "cam")
    counter.update(1, 500, 400, "car", 30.0, "cam")
    # Second crossing of the same arm by the same track should be ignored.
    counter.update(1, 500, 200, "car", 30.0, "cam")
    counter.update(1, 500, 400, "car", 30.0, "cam")
    assert counter.get_summary()["per_arm"]["Utara"]["masuk"] == 1


def test_persistence_called_on_crossing():
    repo = InMemoryDetectionRepository()
    counter = VirtualLineCounter("http://example/stream", repo=repo)
    counter.update(1, 500, 200, "car", 30.0, "cam")
    counter.update(1, 500, 400, "car", 30.0, "cam")
    assert len(repo.crossings) == 1
    assert isinstance(repo.crossings[0], LineCrossingEvent)
    assert repo.crossings[0].direction.startswith("Utara-")

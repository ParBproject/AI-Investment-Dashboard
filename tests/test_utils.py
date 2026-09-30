import numpy as np

from src.utils import format_pct, returns_histogram, weights_pie_chart


def test_format_pct_and_non_finite() -> None:
    assert format_pct(0.0342) == "3.42%"
    assert format_pct(float("nan")) == "n/a"
    assert format_pct(float("inf")) == "n/a"


def test_histogram_survives_a_constant_sample() -> None:
    fig = returns_histogram(np.array([1.0, 1.0, 1.0]))
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"


def test_histogram_adds_a_density_when_the_sample_has_spread() -> None:
    fig = returns_histogram(np.linspace(0.0, 1.0, 50))
    assert any(trace.type == "scatter" for trace in fig.data)


def test_short_weights_use_a_bar_chart() -> None:
    fig = weights_pie_chart(np.array([0.7, -0.2, 0.5]), ["A", "B", "C"])
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].y) == [70.0, -20.0, 50.0]


def test_long_only_weights_stay_a_pie() -> None:
    fig = weights_pie_chart(np.array([0.25, 0.75]), ["A", "B"])
    assert fig.data[0].type == "pie"

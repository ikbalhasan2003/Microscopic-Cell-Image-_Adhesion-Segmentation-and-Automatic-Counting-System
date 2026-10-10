import numpy as np
import pytest
from matplotlib.figure import Figure

from core.analysis import CSV_COLUMNS, CellAnalyzer


@pytest.fixture
def squares():
    lab = np.zeros((100, 120), np.int32)
    lab[60:80, 10:30] = 7   # 400 px
    lab[10:25, 70:85] = 3   # 225 px
    lab[12:17, 20:25] = 9   # 25 px < min_area
    return lab


def test_areas_and_ids(squares):
    cells = CellAnalyzer().run(squares, 50)
    assert [c.id for c in cells] == [1, 2]
    assert [c.area for c in cells] == [225, 400]  # top-right square is read first


def test_renumber_matches_ids(squares):
    a = CellAnalyzer()
    lab = a.renumber(squares, 50)
    for c in a.run(squares, 50):
        assert lab[int(c.centroid[0]), int(c.centroid[1])] == c.id


def test_summary_and_dataframe(squares):
    a = CellAnalyzer()
    cells = a.run(squares, 50)
    s = a.summary(cells)
    assert s["count"] == 2 and s["mean_area"] == 312.5 and s["median_area"] == 312.5
    assert list(a.to_dataframe(cells).columns) == CSV_COLUMNS
    assert a.summary([])["count"] == 0


def test_histogram_figures(squares):
    a = CellAnalyzer()
    assert isinstance(a.histogram([]), Figure)
    assert isinstance(a.histogram(a.run(squares, 50)), Figure)


def test_empty_labels():
    assert CellAnalyzer().run(np.zeros((30, 30), np.int32), 10) == []


@pytest.mark.parametrize("lab", [np.zeros((10, 10)), np.zeros((2, 10, 10), np.int32),
                                 -np.ones((10, 10), np.int32)])
def test_invalid(lab):
    with pytest.raises(ValueError):
        CellAnalyzer().run(lab, 10)

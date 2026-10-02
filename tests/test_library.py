import os
from pathlib import Path

import pytest

from tuxun.search import Filters
from tests.conftest import make_image


def names(results):
    return [Path(r["path"]).name for r in results]


def test_search_ranks_matching_image_first(library, photos):
    library.add_folder(str(photos))
    result = library.run_index()
    assert result.added == 3
    assert [Path(p).name for p in result.failed] == ["broken.jpg"]

    assert names(library.search_text("红", limit=1)) == ["red.jpg"]
    assert names(library.search_text("绿", limit=1)) == ["green.png"]
    assert names(library.search_text("蓝"))[0] == "blue.jpg"


def test_index_is_incremental(library, embedder, photos):
    library.add_folder(str(photos))
    library.run_index()
    calls = embedder.image_calls

    again = library.run_index()
    assert again.added == 0 and embedder.image_calls == calls

    make_image(photos / "red.jpg", (10, 10, 250))
    os.utime(photos / "red.jpg", (1, 1))
    (photos / "blue.jpg").unlink()
    changed = library.run_index()
    assert changed.added == 1 and changed.removed == 1
    assert names(library.search_text("蓝", limit=1)) == ["red.jpg"]


def test_similar_excludes_query_image(library, photos):
    make_image(photos / "red2.jpg", (240, 20, 20))
    library.add_folder(str(photos))
    library.run_index()
    red = library.search_text("红", limit=1)[0]
    similar = library.search_similar(red["id"], limit=2)
    assert red["id"] not in [r["id"] for r in similar]
    assert names(similar)[0] == "red2.jpg"


def test_pagination_and_date_filter(library, photos):
    library.add_folder(str(photos))
    library.run_index()
    all_results = library.search_text("红")
    assert library.search_text("红", limit=1, offset=1) == all_results[1:2]
    assert library.search_text("红", filters=Filters(date_to=0)) == []


def test_folder_rules(library, photos, tmp_path):
    library.add_folder(str(photos))
    with pytest.raises(ValueError):
        library.add_folder(str(photos / "sub"))
    with pytest.raises(ValueError):
        library.add_folder(str(tmp_path))
    with pytest.raises(ValueError):
        library.add_folder(str(tmp_path / "missing"))

    library.run_index()
    library.remove_folder(str(photos.resolve()))
    assert library.db.count() == 0
    assert library.search_text("红") == []


def test_model_change_clears_index(library, embedder, photos):
    library.add_folder(str(photos))
    library.run_index()
    embedder.name = "another-model"
    calls = embedder.image_calls
    assert library.run_index().added == 3
    assert embedder.image_calls == calls + 3

from src.label_schema import label_display, label_items, label_name, load_label_schema


def test_ctspine1k_verse_schema_matches_expected_vertebra_names() -> None:
    schema = load_label_schema()
    assert schema["schema_id"] == "ctspine1k_verse_vertebrae_1_25"
    assert schema["formal_task_locked"] is False
    assert label_name(1) == "C1"
    assert label_name(7) == "C7"
    assert label_name(8) == "T1"
    assert label_name(19) == "T12"
    assert label_name(20) == "L1"
    assert label_name(24) == "L5"
    assert label_name(25) == "L6"


def test_unknown_label_remains_visible_without_inventing_anatomy() -> None:
    assert label_name(99) is None
    assert label_display(99) == "label 99"


def test_label_items_are_sorted_and_background_is_optional() -> None:
    items = label_items([24, 0, 7, 24, 8])
    assert [item["value"] for item in items] == [7, 8, 24]
    assert [item["display"] for item in items] == ["C7 (7)", "T1 (8)", "L5 (24)"]

    with_background = label_items([1, 0], include_background=True)
    assert [item["value"] for item in with_background] == [0, 1]
    assert with_background[0]["display"] == "background (0)"

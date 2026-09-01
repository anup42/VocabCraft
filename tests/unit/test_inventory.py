from vocabcraft.inventory import build_inventory


def test_inventory_covers_every_token(fake_tokenizer: object) -> None:
    records = build_inventory(fake_tokenizer)  # type: ignore[arg-type]
    assert [record.original_id for record in records] == list(range(11))
    assert records[2].is_unknown
    assert records[7].is_extra_id
    assert records[10].unicode_scripts >= {"Latin", "Devanagari"}

from prirucka2024.rst_parser_00 import parse_rst_headers


def test_parse_rst_headers():
    """Test the RST header parsing logic."""
    rst_text = """
    Design Principles
    =================

    Principle 1: Intuitive Construction
    ===================================

    Some content under principle 1.

    Principle 2: Capture Design Intent
    ==================================

    Some content under principle 2.
    """

    parsed = parse_rst_headers(rst_text)

    assert len(parsed) == 3, "Incorrect number of sections parsed."
    assert parsed[0]["header"] == "Design Principles"
    assert parsed[0]["level"] == 1
    assert parsed[0]["parent"] is None
    assert parsed[1]["header"] == "Principle 1: Intuitive Construction"
    assert parsed[1]["parent"] == "Design Principles"
    assert "Some content under principle 1." in parsed[1]["content"]

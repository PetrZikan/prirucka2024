from prirucka2024.rst_parser import parse_rst_headers_with_graph


def test_parse_rst_headers_with_graph():
    """Test the RST header parsing with full graph structure."""
    rst_text = """Header One
==========

Some content under Header One.

Header Two
----------

Header Three
^^^^^^^^^^^^

Some content under Header Three.

Another Root
============

Child of Another Root
----------------------
"""

    parsed = parse_rst_headers_with_graph(rst_text)

    assert len(parsed) == 5, "Incorrect number of sections parsed."
    assert parsed[0]["header"] == "Header One"
    assert parsed[0]["full_path"] == ["Header One"]
    assert parsed[0]["content"] == "Some content under Header One."
    assert parsed[1]["header"] == "Header Two"
    assert parsed[1]["full_path"] == ["Header One", "Header Two"]
    assert parsed[2]["header"] == "Header Three"
    assert parsed[2]["full_path"] == ["Header One", "Header Two", "Header Three"]
    assert parsed[2]["content"] == "Some content under Header Three."
    assert parsed[4]["full_path"] == ["Another Root", "Child of Another Root"]
    assert parsed[4]["parent"] == "Another Root"

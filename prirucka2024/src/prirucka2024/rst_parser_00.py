import re
from typing import List, Dict


def parse_rst_headers(rst_text: str) -> List[Dict]:
    """
    Parse RST headers and their associated content.
    Args:
        rst_text (str): The content of the .rst file.
    Returns:
        List[Dict]: A list of dictionaries containing headers and their metadata.
    """
    # Regex to match headers (e.g., underlined with =, -, ~, etc.)
    header_pattern = r"^(.*)\n([=~`^\"*+#-]+)\n$"
    matches = re.finditer(header_pattern, rst_text, re.MULTILINE)

    sections = []
    previous_index = 0
    header_stack = []

    for match in matches:
        header_text = match.group(1).strip()
        underline_char = match.group(2)[0]
        header_level = determine_header_level(underline_char)

        # Content between this header and the next
        content_start = match.end()
        next_match = next(matches, None)
        content_end = next_match.start() if next_match else len(rst_text)
        content = rst_text[content_start:content_end].strip()

        # Maintain the hierarchy
        while header_stack and header_stack[-1]["level"] >= header_level:
            header_stack.pop()

        # Add this header to the stack
        header_metadata = {
            "header": header_text,
            "level": header_level,
            "content": content,
            "parent": header_stack[-1]["header"] if header_stack else None,
        }
        sections.append(header_metadata)
        header_stack.append(header_metadata)

    return sections


def determine_header_level(underline_char: str) -> int:
    """
    Determine the header level based on the underline character.
    Args:
        underline_char (str): The underline character.
    Returns:
        int: The header level.
    """
    header_characters = ["=", "-", "~", "^", '"', "*", "+", "#"]
    return (
        header_characters.index(underline_char) + 1
        if underline_char in header_characters
        else 10
    )

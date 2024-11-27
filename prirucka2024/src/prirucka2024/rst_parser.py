import re
from typing import List, Dict, Optional

# Define the hierarchy of RST headers
HEADER_ORDER = ["=", "-", "^", '"']


def parse_rst_headers_with_graph(rst_text: str) -> List[Dict]:
    """
    Parse RST headers and maintain a full graph structure for nesting.
    Args:
        rst_text (str): The content of the .rst file.
    Returns:
        List[Dict]: A list of dictionaries representing sections with full paths.
    """
    # Regex to match headers (e.g., underlined with =, -, ^, ")
    # header_pattern = r"^(.*)\n([=~`^\"*+#-]+)\n$"
    header_pattern = r"^\s*(.*?)\s*\n\s*([=~`^\"*+#-]+)\s*$"
    matches = list(re.finditer(header_pattern, rst_text, re.MULTILINE))

    sections = []
    header_stack = []  # Stack to track current path

    for i, match in enumerate(matches):
        header_text = match.group(1).strip()
        underline_char = match.group(2)[0]
        header_level = determine_header_level(underline_char)

        # Extract content between this header and the next
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(rst_text)
        content = rst_text[content_start:content_end].strip()

        # Update the header stack to maintain the hierarchy
        while header_stack and header_stack[-1]["level"] >= header_level:
            header_stack.pop()

        # Build full path
        full_path = [h["header"] for h in header_stack] + [header_text]

        # Store section metadata
        section_metadata = {
            "header": header_text,
            "level": header_level,
            "content": content,
            "full_path": full_path,
            "parent": header_stack[-1]["header"] if header_stack else None,
        }
        sections.append(section_metadata)
        header_stack.append(section_metadata)

    return sections


def determine_header_level(underline_char: str) -> int:
    """
    Determine the header level based on the underline character.
    Args:
        underline_char (str): The underline character.
    Returns:
        int: The header level.
    """
    if underline_char in HEADER_ORDER:
        return HEADER_ORDER.index(underline_char) + 1
    return len(HEADER_ORDER) + 1  # Default to a deeper level if not in the order

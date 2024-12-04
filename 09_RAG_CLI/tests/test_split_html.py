import pickle
import pytest
from prirucka2024.split_html_on_headers import split_html


@pytest.fixture
def setup_test_data(tmp_path):
    """Set up test data directory and files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Create sample HTML files
    sample1 = test_dir / "sample1.html"
    sample1.write_text(
        """
        <html>
          <body>
            <div class="target-class">
              <h1>Header 1</h1>
              <p>Paragraph under header 1</p>
              <h2>Header 2</h2>
              <p>Paragraph under header 2</p>
            </div>
            <div class="other-class">
              <h3>Another Header</h3>
              <p>Content not to be included when using filter.</p>
            </div>
          </body>
        </html>
        """
    )

    return test_dir


def test_split_html_basic(setup_test_data, tmp_path):
    """Test basic splitting of HTML."""
    file_path = setup_test_data / "sample1.html"
    output_pkl = tmp_path / "output.pkl"
    output_txt = tmp_path / "output.txt"

    # Run split_html
    split_html(
        file_path, output_pkl, output_txt, interactive=False, drop_empty_metadata=True
    )

    # Check results
    assert output_pkl.exists()
    assert output_txt.exists()

    with open(output_pkl, "rb") as f:
        splits = pickle.load(f)

    # Verify the number of splits with non-empty metadata
    assert len(splits) == 3  # Two splits with non-empty metadata
    assert "Header 1" in splits[0].metadata
    assert "Paragraph under header 1" in splits[0].page_content
    assert "Header 2" in splits[1].metadata
    assert "Paragraph under header 2" in splits[1].page_content
    assert "Header 3" in splits[2].metadata
    assert "Content not to be included when using filter." in splits[2].page_content


def test_split_html_with_filter(setup_test_data, tmp_path):
    """Test filtering HTML by tag and class."""
    file_path = setup_test_data / "sample1.html"
    output_pkl = tmp_path / "output_filtered.pkl"
    output_txt = tmp_path / "output_filtered.txt"

    # Run split_html with filtering and drop_empty_metadata
    split_html(
        file_path,
        output_pkl,
        output_txt,
        interactive=False,
        target_tag="div",
        target_class="target-class",
        drop_empty_metadata=True,
    )

    # Check results
    assert output_pkl.exists()
    assert output_txt.exists()

    with open(output_pkl, "rb") as f:
        splits = pickle.load(f)

    # Verify the number of splits with non-empty metadata
    assert len(splits) == 2  # Two splits within target-class with non-empty metadata
    assert "Header 1" in splits[0].metadata
    assert "Paragraph under header 1" in splits[0].page_content
    assert "Header 2" in splits[1].metadata
    assert "Paragraph under header 2" in splits[1].page_content


def test_split_html_empty_filter(setup_test_data, tmp_path):
    """Test filtering HTML with no matching tag and class."""
    file_path = setup_test_data / "sample1.html"
    output_pkl = tmp_path / "output_empty.pkl"
    output_txt = tmp_path / "output_empty.txt"

    # Run split_html with non-matching filter
    split_html(
        file_path,
        output_pkl,
        output_txt,
        interactive=False,
        target_tag="div",
        target_class="nonexistent-class",
    )

    # Check results
    assert output_pkl.exists()
    assert output_txt.exists()

    with open(output_pkl, "rb") as f:
        splits = pickle.load(f)

    assert len(splits) == 0  # No splits should match

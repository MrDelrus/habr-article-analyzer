import numpy as np
import pandas as pd
import pytest

from ml_training.data.targets import Target


@pytest.fixture
def sample_dataset_single_labels() -> pd.DataFrame:
    """Create a sample dataset with single string labels."""
    return pd.DataFrame({"category": ["cat", "dog", "cat", "bird", "dog"]})


@pytest.fixture
def sample_dataset_multilabel() -> pd.DataFrame:
    """Create a sample dataset with multi-label lists."""
    return pd.DataFrame(
        {"tags": [["red", "blue"], ["green"], ["red", "green", "blue"], ["blue"], []]}
    )


@pytest.fixture
def sample_dataset_mixed_types() -> pd.DataFrame:
    """Create a sample dataset with mixed single and multi-label entries."""
    return pd.DataFrame(
        {"labels": ["single", ["multi", "label"], "another", ["multi"]]}
    )


# Unit tests


def test_init_with_single_labels(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")

    assert isinstance(target.dataset, pd.DataFrame)
    assert target.column_name == "category"
    assert len(target.targets) == 5
    assert target.labels == ["bird", "cat", "dog"]
    assert target.binary_mask.shape == (5, 3)


def test_init_with_multilabel(sample_dataset_multilabel: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_multilabel, column_name="tags")

    assert isinstance(target.targets, pd.Series)
    assert all(isinstance(x, list) for x in target.targets)
    assert target.labels == ["blue", "green", "red"]
    assert target.binary_mask.shape == (5, 3)


def test_init_with_mixed_types(sample_dataset_mixed_types: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_mixed_types, column_name="labels")

    assert all(isinstance(x, list) for x in target.targets)
    assert target.labels == sorted(["another", "multi", "single", "label"])
    assert target.binary_mask.shape == (4, 4)


def test_get_labels_single_categories() -> None:
    targets = pd.Series([["cat"], ["dog"], ["cat"]])
    labels = Target._get_labels(targets)

    assert labels == ["cat", "dog"]
    assert isinstance(labels, list)


def test_get_labels_multilabel() -> None:
    targets = pd.Series([["red", "blue"], ["green"], ["red", "green"]])
    labels = Target._get_labels(targets)

    assert set(labels) == {"red", "blue", "green"}
    assert len(labels) == 3


def test_get_labels_empty_lists() -> None:
    targets = pd.Series([[], ["cat"], []])
    labels = Target._get_labels(targets)

    assert labels == ["cat"]


def test_as_str_list_with_strings() -> None:
    targets = pd.Series(["cat", "dog", "bird"])
    result = Target._as_str_list(targets)

    assert all(isinstance(x, list) for x in result)
    assert result.iloc[0] == ["cat"]
    assert result.iloc[1] == ["dog"]


def test_as_str_list_with_lists() -> None:
    targets = pd.Series([["cat", "dog"], ["bird"]])
    result = Target._as_str_list(targets)

    assert all(isinstance(x, list) for x in result)
    assert result.iloc[0] == ["cat", "dog"]
    assert result.iloc[1] == ["bird"]


def test_as_str_list_mixed() -> None:
    targets = pd.Series(["cat", ["dog", "bird"], "fish"])
    result = Target._as_str_list(targets)

    assert all(isinstance(x, list) for x in result)
    assert result.iloc[0] == ["cat"]
    assert result.iloc[1] == ["dog", "bird"]
    assert result.iloc[2] == ["fish"]


def test_label_to_id_existing_label(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")

    assert target.label_to_id("cat") == 1
    assert target.label_to_id("dog") == 2
    assert target.label_to_id("bird") == 0


def test_label_to_id_nonexistent_label_raises(
    sample_dataset_single_labels: pd.DataFrame,
) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")

    with pytest.raises(KeyError, match="nonexistent not in labels"):
        target.label_to_id("nonexistent")


def test_len(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")

    assert len(target) == 3


def test_getitem_with_string_label(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")
    result = target["cat"]

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.int64
    assert np.array_equal(result, [1, 0, 1, 0, 0])


def test_getitem_with_int_index(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")
    result = target[1]  # should be "cat"

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert np.array_equal(result, [1, 0, 1, 0, 0])


def test_getitem_with_list_of_strings(sample_dataset_multilabel: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_multilabel, column_name="tags")
    result = target[["red", "blue"]]

    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 2)
    # Check that we get both red and blue columns


def test_getitem_with_list_of_ints(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")
    result = target[[0, 1]]  # bird and cat

    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 2)


def test_getitem_with_numpy_array(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")
    result = target[np.array([0, 2])]  # bird and dog

    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 2)


def test_get_coverage_all_labels(sample_dataset_multilabel: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_multilabel, column_name="tags")
    coverage = target.get_coverage()

    assert isinstance(coverage, float)
    assert 0 <= coverage <= 1
    assert coverage == 0.8


def test_get_sizes_subset_labels(sample_dataset_single_labels: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_single_labels, column_name="category")
    sizes = target.get_sizes(["cat", "dog"])

    assert isinstance(sizes, np.ndarray)
    assert np.array_equal(sizes, [2, 2])


def test_binary_mask_correctness(sample_dataset_multilabel: pd.DataFrame) -> None:
    target = Target(dataset=sample_dataset_multilabel, column_name="tags")

    # Check shape
    assert target.binary_mask.shape == (5, 3)

    # Check that sum along rows equals number of labels per sample
    row_sums = target.binary_mask.sum(axis=1)
    expected_sums = [2, 1, 3, 1, 0]  # Based on our sample data
    assert np.array_equal(row_sums, expected_sums)


def test_empty_dataset() -> None:
    empty_df = pd.DataFrame({"labels": []})
    target = Target(dataset=empty_df, column_name="labels")

    assert len(target) == 0
    assert target.labels == []
    assert target.binary_mask.shape == (0, 0)

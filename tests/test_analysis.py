import numpy as np
import pytest

from ffprime.electrostatics.analysis import (
    project_field,
    cosine_similarity,
    rms_deviation,
)


def test_project_field_x_axis():
    """Project a field onto the x-axis."""
    field = np.array([3.0, 4.0, 5.0])
    direction = np.array([1.0, 0.0, 0.0])

    result = project_field(field, direction)

    assert np.isclose(result, 3.0)


def test_project_field_y_axis():
    """Project a field onto the y-axis."""
    field = np.array([3.0, 4.0, 5.0])
    direction = np.array([0.0, 1.0, 0.0])

    result = project_field(field, direction)

    assert np.isclose(result, 4.0)


def test_project_field_z_axis():
    """Project a field onto the z-axis."""
    field = np.array([3.0, 4.0, 5.0])
    direction = np.array([0.0, 0.0, 1.0])

    result = project_field(field, direction)

    assert np.isclose(result, 5.0)


def test_project_field_normalizes_direction():
    """Projection should be independent of the direction vector magnitude."""
    field = np.array([3.0, 4.0, 5.0])

    result1 = project_field(field, np.array([1.0, 0.0, 0.0]))
    result2 = project_field(field, np.array([10.0, 0.0, 0.0]))

    assert np.isclose(result1, result2)


def test_project_multiple_fields():
    """Project several field vectors at once."""
    fields = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    direction = np.array([0.0, 0.0, 1.0])

    result = project_field(fields, direction)

    expected = np.array([3.0, 6.0, 9.0])

    assert np.allclose(result, expected)


def test_project_field_diagonal():
    """Projection onto a diagonal direction."""
    field = np.array([1.0, 1.0, 0.0])
    direction = np.array([1.0, 1.0, 0.0])

    result = project_field(field, direction)

    assert np.isclose(result, np.sqrt(2.0))


def test_project_field_zero_direction():
    """A zero direction vector should raise an error."""
    field = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        project_field(field, np.array([0.0, 0.0, 0.0]))


def test_cosine_similarity_identical():
    field1 = np.array([[1.0, 2.0, 3.0]])
    field2 = np.array([[1.0, 2.0, 3.0]])

    result = cosine_similarity(field1, field2)

    assert np.isclose(result, 1.0)


def test_cosine_similarity_opposite():
    field1 = np.array([[1.0, 0.0, 0.0]])
    field2 = np.array([[-1.0, 0.0, 0.0]])

    result = cosine_similarity(field1, field2)

    assert np.isclose(result, -1.0)


def test_cosine_similarity_orthogonal():
    field1 = np.array([[1.0, 0.0, 0.0]])
    field2 = np.array([[0.0, 1.0, 0.0]])

    result = cosine_similarity(field1, field2)

    assert np.isclose(result, 0.0)


def test_cosine_similarity_shape_error():
    field1 = np.array([[1.0, 2.0, 3.0]])
    field2 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        cosine_similarity(field1, field2)


def test_cosine_similarity_zero_field():
    field1 = np.zeros((1, 3))
    field2 = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        cosine_similarity(field1, field2)


def test_rms_deviation_identical():
    field1 = np.array([[1.0, 2.0, 3.0]])
    field2 = np.array([[1.0, 2.0, 3.0]])

    result = rms_deviation(field1, field2)

    assert np.isclose(result, 0.0)


def test_rms_deviation_known_value():
    field1 = np.array([[1.0, 2.0, 3.0]])
    field2 = np.array([[2.0, 3.0, 5.0]])

    result = rms_deviation(field1, field2)

    expected = np.sqrt((1 + 1 + 4) / 3)

    assert np.isclose(result, expected)


def test_rms_deviation_multiple_fields():
    field1 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    field2 = np.array(
        [
            [2.0, 2.0, 4.0],
            [5.0, 5.0, 7.0],
        ]
    )

    result = rms_deviation(field1, field2)

    expected = np.sqrt(2.0 / 3.0)

    assert np.isclose(result, expected)


def test_rms_deviation_shape_error():
    field1 = np.array([[1.0, 2.0, 3.0]])
    field2 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        rms_deviation(field1, field2)
import pytest
import numpy as np
from forgetful_ml import engagement_score

def test_engagement_score_positive():
    """
    Test a simple case where retain set (r) and forget set (f) have distinct means.
    """
    r = np.array([[1.0, 2.0], [2.0, 1.0]])  # retain set
    f = np.array([[10.0, 10.0], [11.0, 11.0]])  # forget set
    
    score = engagement_score(r, f)
    assert score > 0.0, "Engagement score should be positive for distinct sets."


def test_engagement_score_same_sets():
    """
    Test a scenario where r and f are the same set. The engagement score should raise a ValueError.
    """
    r = np.array([[1.0, 2.0], [2.0, 1.0]])
    f = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="Denominator is zero"):
        engagement_score(r, f)

def test_engagement_score_single_dimension():
    """
    Test with single-dimensional data.
    """
    r = np.array([[1], [2], [3]])  # shape (3, 1)
    f = np.array([[3], [4], [5]])  # shape (3, 1)

    score = engagement_score(r, f)
    assert score > 0.0, "Single-dimensional data should produce a positive score here."

def test_engagement_score_denominator_zero():
    """
    Test a scenario where the denominator would be zero, causing a ValueError.
    This can happen if r and f have the exact same mean as the combined mean.
    For example, if both sets have identical data points.
    """
    r = np.array([[1.0, 1.0], [1.0, 1.0]])
    f = np.array([[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="Denominator is zero"):
        engagement_score(r, f)


def test_engagement_score_different_shapes():
    """
    Test that the function works when r and f have different numbers of samples but the same dimension d.
    """
    r = np.array([[0, 0], [1, 1]])     # shape (2, 2)
    f = np.array([[2, 2], [3, 3], [4, 4]])  # shape (3, 2)

    score = engagement_score(r, f)
    assert isinstance(score, float), "Should compute a float for different shaped inputs."


def test_engagement_score_negative_values():
    """
    Test that the function can handle negative values. Variance calculations should be unaffected.
    """
    r = np.array([[-1, -2], [-3, -4]])
    f = np.array([[1, 2], [3, 4]])

    score = engagement_score(r, f)
    assert score >= 0.0, "Engagement score should be non-negative."
    
def test_the_correctness_of_value():
    """
    Test the case where r and f have distinct means and variances.
    """
    r = np.array([[4.0, 2.0], [0.0, 2.0]])
    f = np.array([[10.0, 15.0], [6.0, 1.0]])
    
    score = engagement_score(r, f)
    
    assert score == pytest.approx(3.16666666666666), "Engagement score should be 3.16666666666666."
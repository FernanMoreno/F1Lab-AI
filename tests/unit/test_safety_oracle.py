"""Test module for SafetyOracle implementation."""

import pytest
from reglabsim.safety.safety_oracle import SafetyOracle


def test_safety_oracle():
    """Test the SafetyOracle implementation."""
    # Create a safety oracle instance
    _oracle = SafetyOracle()
    
    # Test that the object can be created
    assert _oracle is not None


if __name__ == "__main__":
    test_safety_oracle()
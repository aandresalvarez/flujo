"""Tests for flujo.application.temperature module."""

import pytest
from unittest.mock import patch

from flujo.application.temperature import temp_for_round


class TestTempForRound:
    """Test temp_for_round function."""

    def test_temp_for_round_within_schedule(self):
        """Test temp_for_round when round is within schedule length."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0.1, 0.2, 0.3, 0.4, 0.5]

            # Test rounds within schedule
            assert temp_for_round(0) == 0.1
            assert temp_for_round(1) == 0.2
            assert temp_for_round(2) == 0.3
            assert temp_for_round(3) == 0.4
            assert temp_for_round(4) == 0.5

    def test_temp_for_round_beyond_schedule(self):
        """Test temp_for_round when round is beyond schedule length."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0.1, 0.2, 0.3]

            # Test rounds beyond schedule (should use last value)
            assert temp_for_round(3) == 0.3
            assert temp_for_round(4) == 0.3
            assert temp_for_round(10) == 0.3
            assert temp_for_round(100) == 0.3

    def test_temp_for_round_single_value_schedule(self):
        """Test temp_for_round with single value schedule."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0.5]

            # Test with single value schedule
            assert temp_for_round(0) == 0.5
            assert temp_for_round(1) == 0.5
            assert temp_for_round(5) == 0.5

    def test_temp_for_round_empty_schedule(self):
        """Test temp_for_round with empty schedule (edge case)."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = []

            # Test with empty schedule (should raise IndexError)
            with pytest.raises(IndexError):
                temp_for_round(0)

    def test_temp_for_round_negative_round(self):
        """Test temp_for_round with negative round number."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0.1, 0.2, 0.3]

            # Negative indices within bounds
            assert temp_for_round(-1) == 0.3  # Last element
            assert temp_for_round(-2) == 0.2  # Second to last element
            assert temp_for_round(-3) == 0.1  # First element
            # Out of bounds negative index
            with pytest.raises(IndexError):
                temp_for_round(-4)
            with pytest.raises(IndexError):
                temp_for_round(-5)

    def test_temp_for_round_float_temperatures(self):
        """Test temp_for_round with float temperature values."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0.0, 0.1, 0.5, 1.0]

            assert temp_for_round(0) == 0.0
            assert temp_for_round(1) == 0.1
            assert temp_for_round(2) == 0.5
            assert temp_for_round(3) == 1.0
            assert temp_for_round(4) == 1.0  # Beyond schedule

    def test_temp_for_round_integer_temperatures(self):
        """Test temp_for_round with integer temperature values."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [1, 2, 3]

            assert temp_for_round(0) == 1
            assert temp_for_round(1) == 2
            assert temp_for_round(2) == 3
            assert temp_for_round(3) == 3  # Beyond schedule

    def test_temp_for_round_mixed_temperatures(self):
        """Test temp_for_round with mixed temperature types."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            mock_settings.t_schedule = [0, 0.5, 1, 1.5]

            assert temp_for_round(0) == 0
            assert temp_for_round(1) == 0.5
            assert temp_for_round(2) == 1
            assert temp_for_round(3) == 1.5
            assert temp_for_round(4) == 1.5  # Beyond schedule

    def test_temp_for_round_large_schedule(self):
        """Test temp_for_round with large schedule."""
        with patch("flujo.application.temperature.settings") as mock_settings:
            large_schedule = [i * 0.1 for i in range(100)]
            mock_settings.t_schedule = large_schedule

            # Test various rounds
            assert temp_for_round(0) == 0.0
            assert temp_for_round(50) == 5.0
            assert temp_for_round(99) == 9.9
            assert temp_for_round(100) == 9.9  # Beyond schedule
            assert temp_for_round(200) == 9.9  # Far beyond schedule

"""
Comprehensive tests for configuration system and MPP selection logic.

This module tests:
- MPP selection in fixed and auto modes
- Configuration validation and overrides
- Config loading and preset inheritance
- Integration between config and MPP selection
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from histoseg_plugin.tiling.config_ops import load_config_with_presets
from histoseg_plugin.tiling.parameter_models import Config, Tiling
from histoseg_plugin.tiling.process_wsi import (
    _bounds,
    _get_level_mpps,
    select_tile_level_auto,
    select_tile_level_fixed,
    select_tile_level_from_config,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_wsi_standard():
    """Mock WSI with standard pyramid levels and MPP metadata."""
    mock_wsi_obj = Mock()
    mock_openslide = Mock()

    # Standard pyramid: level 0 = 0.25 μm/px, each level doubles
    mock_openslide.level_count = 4
    mock_openslide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
    mock_openslide.properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

    mock_wsi_obj.getOpenSlide.return_value = mock_openslide
    return mock_wsi_obj


@pytest.fixture
def mock_wsi_no_mpp():
    """Mock WSI without MPP metadata (fallback to downsamples)."""
    mock_wsi_obj = Mock()
    mock_openslide = Mock()

    mock_openslide.level_count = 3
    mock_openslide.level_downsamples = [1.0, 4.0, 16.0]
    mock_openslide.properties = {}  # No MPP metadata

    mock_wsi_obj.getOpenSlide.return_value = mock_openslide
    return mock_wsi_obj


@pytest.fixture
def mock_wsi_single_level():
    """Mock WSI with only one pyramid level."""
    mock_wsi_obj = Mock()
    mock_openslide = Mock()

    mock_openslide.level_count = 1
    mock_openslide.level_downsamples = [1.0]
    mock_openslide.properties = {"openslide.mpp-x": "0.50", "openslide.mpp-y": "0.50"}

    mock_wsi_obj.getOpenSlide.return_value = mock_openslide
    return mock_wsi_obj


@pytest.fixture
def sample_configs():
    """Sample configurations for testing."""
    return {
        "fixed_mode": Config(
            tiling=Tiling(
                level_mode="fixed",
                tile_level=1,
                tile_size=256,
                step_size=256,
                # target_tile_mpp will default to 0.50, but in fixed mode it's used for tolerance check
            )
        ),
        "auto_mode": Config(
            tiling=Tiling(
                level_mode="auto",
                target_tile_mpp=0.50,
                mpp_tolerance=0.10,
                level_policy="closest",
                tile_size=256,
                step_size=256,
            )
        ),
        "auto_lower": Config(
            tiling=Tiling(
                level_mode="auto",
                target_tile_mpp=0.75,
                mpp_tolerance=0.05,
                level_policy="lower",
                tile_size=512,
                step_size=512,
            )
        ),
    }


@pytest.fixture
def temp_yaml_configs():
    """Create temporary YAML config files for testing."""
    configs = {}

    # Default config
    default_config = {
        "tiling": {
            "level_mode": "auto",
            "tile_size": 256,
            "step_size": 256,
            "target_tile_mpp": 0.30,
            "mpp_tolerance": 0.1,
            "level_policy": "closest",
        },
        "seg_params": {"sthresh": 8},
        "filter_params": {"a_t": 100},
    }

    # Preset configs
    biopsy_preset = {"seg_params": {"sthresh": 15}, "filter_params": {"a_t": 50}}

    fixed_preset = {"tiling": {"level_mode": "fixed", "tile_level": 2}}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write config files
        default_path = tmpdir / "default.yaml"
        with open(default_path, "w") as f:
            yaml.safe_dump(default_config, f)
        configs["default"] = default_path

        biopsy_path = tmpdir / "biopsy.yaml"
        with open(biopsy_path, "w") as f:
            yaml.safe_dump(biopsy_preset, f)
        configs["biopsy"] = biopsy_path

        fixed_path = tmpdir / "fixed.yaml"
        with open(fixed_path, "w") as f:
            yaml.safe_dump(fixed_preset, f)
        configs["fixed"] = fixed_path

        yield configs


# =============================================================================
# Test MPP Level Extraction
# =============================================================================


class TestMPPExtraction:
    """Test _get_level_mpps function."""

    def test_extract_with_mpp_metadata(self, mock_wsi_standard):
        """Should extract MPPs from metadata when available."""
        mpps = _get_level_mpps(mock_wsi_standard)

        expected = [0.25, 0.50, 1.0, 2.0]  # base * downsamples
        assert mpps == expected

    def test_extract_without_mpp_metadata(self, mock_wsi_no_mpp):
        """Should fallback to downsamples when no MPP metadata."""
        mpps = _get_level_mpps(mock_wsi_no_mpp)

        expected = [1.0, 4.0, 16.0]  # just downsamples
        assert mpps == expected

    def test_extract_invalid_mpp_metadata(self):
        """Should handle corrupted MPP metadata gracefully."""
        mock_wsi_obj = Mock()
        mock_openslide = Mock()

        mock_openslide.level_count = 2
        mock_openslide.level_downsamples = [1.0, 2.0]
        mock_openslide.properties = {
            "openslide.mpp-x": "invalid",
            "openslide.mpp-y": "0.25",
        }

        mock_wsi_obj.getOpenSlide.return_value = mock_openslide

        mpps = _get_level_mpps(mock_wsi_obj)
        assert mpps == [1.0, 2.0]  # fallback to downsamples


# =============================================================================
# Test Fixed Mode MPP Selection
# =============================================================================


class TestMPPSelectionFixed:
    """Test select_tile_level_fixed function."""

    def test_fixed_valid_level(self, mock_wsi_standard):
        """Should return exact level when valid."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_standard, tile_level=1
        )

        assert level == 1
        assert mpp == 0.50  # level 1 MPP
        assert within_tolerance is True
        assert reason == "fixed level; no target MPP"

    def test_fixed_level_clamping_high(self, mock_wsi_standard):
        """Should clamp level to maximum available."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_standard,
            tile_level=10,  # > max level 3
        )

        assert level == 3  # clamped to max
        assert mpp == 2.0  # level 3 MPP
        assert within_tolerance is True

    def test_fixed_level_clamping_low(self, mock_wsi_standard):
        """Should clamp negative level to 0."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_standard, tile_level=-1
        )

        assert level == 0  # clamped to min
        assert mpp == 0.25  # level 0 MPP
        assert within_tolerance is True

    def test_fixed_with_tolerance_check_pass(self, mock_wsi_standard):
        """Should check tolerance when target MPP provided."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_standard,
            tile_level=1,
            target_tile_mpp=0.50,  # exact match
            mpp_tolerance=0.10,
        )

        assert level == 1
        assert mpp == 0.50
        assert within_tolerance is True
        assert reason == "within tolerance"

    def test_fixed_with_tolerance_check_fail(self, mock_wsi_standard):
        """Should fail tolerance check when MPP too different."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_standard,
            tile_level=2,  # mpp = 1.0
            target_tile_mpp=0.50,  # too different
            mpp_tolerance=0.10,
        )

        assert level == 2
        assert mpp == 1.0
        assert within_tolerance is False
        assert "not in [0.450, 0.550]" in reason

    def test_fixed_single_level_wsi(self, mock_wsi_single_level):
        """Should handle single-level WSI correctly."""
        level, mpp, within_tolerance, reason = select_tile_level_fixed(
            mock_wsi_single_level,
            tile_level=5,  # way higher than available
        )

        assert level == 0  # only level available
        assert mpp == 0.50


# =============================================================================
# Test Auto Mode MPP Selection
# =============================================================================


class TestMPPSelectionAuto:
    """Test select_tile_level_auto function."""

    def test_auto_closest_policy(self, mock_wsi_standard):
        """Should select closest level to target MPP."""
        # Target 0.6 should be closest to level 1 (0.5) vs level 2 (1.0)
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.60,
            mpp_tolerance=0.20,
            level_policy="closest",
        )

        assert level == 1
        assert mpp == 0.50
        assert within_tolerance is True

    def test_auto_lower_policy_available(self, mock_wsi_standard):
        """Should select highest resolution level <= target."""
        # Target 0.75, available: [0.25, 0.50, 1.0, 2.0]
        # Lower levels: 0.25, 0.50 -> should pick 0.50 (closest to target)
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.75,
            mpp_tolerance=0.50,
            level_policy="lower",
        )

        assert level == 1  # 0.50 mpp
        assert mpp == 0.50
        assert within_tolerance is True

    def test_auto_lower_policy_fallback(self, mock_wsi_standard):
        """Should fallback to closest when no lower levels available."""
        # Target 0.20, all levels are higher -> should fallback to closest (level 0)
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.20,
            mpp_tolerance=0.10,
            level_policy="lower",
        )

        assert level == 0  # closest available
        assert mpp == 0.25

    def test_auto_higher_policy_available(self, mock_wsi_standard):
        """Should select lowest resolution level >= target."""
        # Target 0.75, available: [0.25, 0.50, 1.0, 2.0]
        # Higher levels: 1.0, 2.0 -> should pick 1.0 (closest to target)
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.75,
            mpp_tolerance=0.50,
            level_policy="higher",
        )

        assert level == 2  # 1.0 mpp
        assert mpp == 1.0
        assert within_tolerance is True

    def test_auto_higher_policy_fallback(self, mock_wsi_standard):
        """Should fallback to closest when no higher levels available."""
        # Target 3.0, all levels are lower -> should fallback to closest (level 3)
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=3.0,
            mpp_tolerance=0.50,
            level_policy="higher",
        )

        assert level == 3  # closest available
        assert mpp == 2.0

    def test_auto_tolerance_failure(self, mock_wsi_standard):
        """Should report tolerance failure with detailed reason."""
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.50,
            mpp_tolerance=0.01,  # very strict
            level_policy="closest",
        )

        assert level == 1  # still selects best level
        assert mpp == 0.50
        assert within_tolerance is True  # 0.50 is exactly target

        # Test actual failure case
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.35,  # between levels
            mpp_tolerance=0.01,
            level_policy="closest",
        )

        assert within_tolerance is False
        assert "target 0.350" in reason


# =============================================================================
# Test Config Dispatcher
# =============================================================================


class TestMPPSelectionDispatcher:
    """Test select_tile_level_from_config function."""

    def test_dispatcher_fixed_mode(self, mock_wsi_standard, sample_configs):
        """Should route to fixed mode function."""
        config = sample_configs["fixed_mode"]

        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, config.tiling
        )

        assert level == 1  # config specifies tile_level=1
        assert mpp == 0.50
        assert (
            "within tolerance" in reason
        )  # level 1 has mpp=0.50, matches target_tile_mpp=0.50

    def test_dispatcher_fixed_mode_no_target(self, mock_wsi_standard):
        """Should route to fixed mode with no target MPP check."""
        # Create a mock tiling object that bypasses Pydantic validation
        from unittest.mock import Mock

        mock_tiling = Mock()
        mock_tiling.level_mode = "fixed"
        mock_tiling.tile_level = 2
        mock_tiling.target_tile_mpp = None
        mock_tiling.mpp_tolerance = 0.10

        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, mock_tiling
        )

        assert level == 2
        assert mpp == 1.0
        assert "fixed level; no target MPP" in reason

    def test_dispatcher_auto_mode(self, mock_wsi_standard, sample_configs):
        """Should route to auto mode function."""
        config = sample_configs["auto_mode"]

        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, config.tiling
        )

        assert level == 1  # closest to target 0.50
        assert mpp == 0.50
        assert within_tolerance is True

    def test_dispatcher_invalid_auto_config(self, mock_wsi_standard):
        """Should raise error for invalid auto config."""
        # Note: Pydantic validation catches this before we even get to the dispatcher
        with pytest.raises(ValueError, match="Auto mode requires target_tile_mpp"):
            Tiling(
                level_mode="auto",
                target_tile_mpp=None,  # Invalid for auto mode
            )


# =============================================================================
# Test Configuration Validation
# =============================================================================


class TestConfigValidation:
    """Test Pydantic model validation for Tiling class."""

    def test_fixed_mode_valid(self):
        """Valid fixed mode config should pass validation."""
        tiling = Tiling(level_mode="fixed", tile_level=2, tile_size=256, step_size=256)

        assert tiling.level_mode == "fixed"
        assert tiling.tile_level == 2

    def test_fixed_mode_invalid_level(self):
        """Fixed mode with negative tile_level should fail validation."""
        with pytest.raises(ValueError, match="Fixed mode requires tile_level"):
            Tiling(
                level_mode="fixed",
                tile_level=-1,  # Invalid
            )

    def test_auto_mode_valid(self):
        """Valid auto mode config should pass validation."""
        tiling = Tiling(level_mode="auto", target_tile_mpp=0.50, level_policy="closest")

        assert tiling.level_mode == "auto"
        assert tiling.target_tile_mpp == 0.50
        assert tiling.tile_level == -1  # Should be ignored in auto mode

    def test_auto_mode_invalid_mpp(self):
        """Auto mode without target_tile_mpp should fail validation."""
        with pytest.raises(ValueError, match="Auto mode requires target_tile_mpp"):
            Tiling(
                level_mode="auto",
                target_tile_mpp=None,  # Invalid
            )

    def test_tile_size_validation(self):
        """Should validate tile_size > 0."""
        with pytest.raises(ValueError):
            Tiling(tile_size=0)

        with pytest.raises(ValueError):
            Tiling(tile_size=-1)

    def test_mpp_tolerance_validation(self):
        """Should validate mpp_tolerance bounds."""
        with pytest.raises(ValueError):
            Tiling(mpp_tolerance=-0.1)  # < 0

        with pytest.raises(ValueError):
            Tiling(mpp_tolerance=1.5)  # > 1


# =============================================================================
# Test Configuration Overrides
# =============================================================================


class TestConfigOverrides:
    """Test Config.with_tiling_overrides method."""

    def test_geometry_only_override(self, sample_configs):
        """Should update geometry without changing mode."""
        config = sample_configs["auto_mode"]

        new_config = config.with_tiling_overrides(tile_size=512, step_size=512)

        # Mode should be preserved
        assert new_config.tiling.level_mode == "auto"
        assert new_config.tiling.target_tile_mpp == 0.50

        # Geometry should be updated
        assert new_config.tiling.tile_size == 512
        assert new_config.tiling.step_size == 512

    def test_explicit_mode_switch_auto_to_fixed(self, sample_configs):
        """Should explicitly switch from auto to fixed mode."""
        config = sample_configs["auto_mode"]

        new_config = config.with_tiling_overrides(level_mode="fixed", tile_level=3)

        assert new_config.tiling.level_mode == "fixed"
        assert new_config.tiling.tile_level == 3

    def test_explicit_mode_switch_fixed_to_auto(self, sample_configs):
        """Should explicitly switch from fixed to auto mode."""
        config = sample_configs["fixed_mode"]

        new_config = config.with_tiling_overrides(
            level_mode="auto", target_tile_mpp=0.25, level_policy="lower"
        )

        assert new_config.tiling.level_mode == "auto"
        assert new_config.tiling.target_tile_mpp == 0.25
        assert new_config.tiling.level_policy == "lower"
        assert new_config.tiling.tile_level == -1  # Reset for auto mode

    def test_implicit_mode_switch_via_tile_level(self, sample_configs):
        """Should infer fixed mode when tile_level provided."""
        config = sample_configs["auto_mode"]

        new_config = config.with_tiling_overrides(tile_level=2)

        assert new_config.tiling.level_mode == "fixed"
        assert new_config.tiling.tile_level == 2

    def test_implicit_mode_switch_via_target_mpp(self, sample_configs):
        """Should infer auto mode when target_tile_mpp provided."""
        config = sample_configs["fixed_mode"]

        new_config = config.with_tiling_overrides(
            target_tile_mpp=0.75, level_policy="higher"
        )

        assert new_config.tiling.level_mode == "auto"
        assert new_config.tiling.target_tile_mpp == 0.75
        assert new_config.tiling.level_policy == "higher"

    def test_override_error_fixed_without_level(self, sample_configs):
        """Should raise error when switching to fixed without tile_level."""
        config = sample_configs["auto_mode"]

        with pytest.raises(ValueError, match="requires a non-negative tile_level"):
            config.with_tiling_overrides(level_mode="fixed")

    def test_override_error_auto_without_mpp(self, sample_configs):
        """Should raise error when switching to auto without target_tile_mpp."""
        config = sample_configs["fixed_mode"]

        with pytest.raises(ValueError, match="requires target_tile_mpp"):
            config.with_tiling_overrides(level_mode="auto")

    def test_convenience_to_fixed(self, sample_configs):
        """Should use convenience method to_fixed."""
        config = sample_configs["auto_mode"]

        new_config = config.to_fixed(tile_level=2)

        assert new_config.tiling.level_mode == "fixed"
        assert new_config.tiling.tile_level == 2

    def test_convenience_to_auto(self, sample_configs):
        """Should use convenience method to_auto."""
        config = sample_configs["fixed_mode"]

        new_config = config.to_auto(target_tile_mpp=0.40, level_policy="higher")

        assert new_config.tiling.level_mode == "auto"
        assert new_config.tiling.target_tile_mpp == 0.40
        assert new_config.tiling.level_policy == "higher"


# =============================================================================
# Test Configuration Loading
# =============================================================================


class TestConfigLoading:
    """Test load_config_with_presets function."""

    def test_load_default_config(self, temp_yaml_configs):
        """Should load default config correctly."""
        config = load_config_with_presets(temp_yaml_configs["default"])

        assert config.tiling.level_mode == "auto"
        assert config.tiling.target_tile_mpp == 0.30
        assert config.seg_params.sthresh == 8

    def test_load_with_preset_merge(self, temp_yaml_configs):
        """Should merge preset over default config."""
        config = load_config_with_presets(
            temp_yaml_configs["default"], presets=[temp_yaml_configs["biopsy"]]
        )

        # Preset should override seg_params
        assert config.seg_params.sthresh == 15
        assert config.filter_params.a_t == 50

        # Default values should be preserved
        assert config.tiling.target_tile_mpp == 0.30
        assert config.tiling.level_mode == "auto"

    def test_load_with_mode_changing_preset(self, temp_yaml_configs):
        """Should handle preset that changes level mode."""
        config = load_config_with_presets(
            temp_yaml_configs["default"], presets=[temp_yaml_configs["fixed"]]
        )

        # Mode should be switched by preset
        assert config.tiling.level_mode == "fixed"
        assert config.tiling.tile_level == 2

        # Other values should be preserved
        assert config.tiling.tile_size == 256

    def test_load_multiple_presets(self, temp_yaml_configs):
        """Should apply multiple presets in order."""
        config = load_config_with_presets(
            temp_yaml_configs["default"],
            presets=[temp_yaml_configs["biopsy"], temp_yaml_configs["fixed"]],
        )

        # Should have both biopsy settings and fixed mode
        assert config.seg_params.sthresh == 15  # from biopsy
        assert config.tiling.level_mode == "fixed"  # from fixed
        assert config.tiling.tile_level == 2


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestConfigIntegration:
    """End-to-end integration tests."""

    def test_fixed_mode_pipeline(self, mock_wsi_standard, sample_configs):
        """Test complete fixed mode pipeline."""
        config = sample_configs["fixed_mode"]

        # Should select fixed level
        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, config.tiling
        )

        assert level == 1
        assert config.tiling.level_mode == "fixed"
        assert (
            "within tolerance" in reason
        )  # level 1 has mpp=0.50, matches default target_tile_mpp=0.50

    def test_auto_mode_pipeline(self, mock_wsi_standard, sample_configs):
        """Test complete auto mode pipeline."""
        config = sample_configs["auto_lower"]

        # Should select based on policy
        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, config.tiling
        )

        # Target 0.75, policy "lower" -> should pick level 1 (0.50)
        assert level == 1
        assert mpp == 0.50
        assert config.tiling.level_policy == "lower"

    def test_config_override_then_selection(self, mock_wsi_standard, sample_configs):
        """Test overriding config then using for selection."""
        config = sample_configs["auto_mode"]

        # Override to fixed mode
        fixed_config = config.with_tiling_overrides(tile_level=3)

        # Should now use fixed selection
        level, mpp, within_tolerance, reason = select_tile_level_from_config(
            mock_wsi_standard, fixed_config.tiling
        )

        assert level == 3
        assert fixed_config.tiling.level_mode == "fixed"


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_wsi_no_levels(self):
        """Should handle WSI with no pyramid levels."""
        mock_wsi_obj = Mock()
        mock_openslide = Mock()
        mock_openslide.level_count = 0
        mock_openslide.level_downsamples = []
        mock_openslide.properties = {}
        mock_wsi_obj.getOpenSlide.return_value = mock_openslide

        with pytest.raises(ValueError, match="No pyramid levels"):
            select_tile_level_fixed(mock_wsi_obj, tile_level=0)

        with pytest.raises(ValueError, match="No pyramid levels"):
            select_tile_level_auto(
                mock_wsi_obj,
                target_tile_mpp=0.50,
                mpp_tolerance=0.1,
                level_policy="closest",
            )

    def test_zero_tolerance(self, mock_wsi_standard):
        """Should handle zero tolerance correctly."""
        # Exact match should pass
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.50,  # exact match with level 1
            mpp_tolerance=0.0,
            level_policy="closest",
        )

        assert within_tolerance is True

        # Non-exact match should fail
        level, mpp, within_tolerance, reason = select_tile_level_auto(
            mock_wsi_standard,
            target_tile_mpp=0.49,  # close but not exact
            mpp_tolerance=0.0,
            level_policy="closest",
        )

        assert within_tolerance is False

    def test_bounds_calculation(self):
        """Test _bounds helper function."""
        lo, hi = _bounds(1.0, 0.1)
        assert lo == 0.9
        assert hi == 1.1

        lo, hi = _bounds(0.5, 0.2)
        assert lo == 0.4
        assert hi == 0.6


if __name__ == "__main__":
    pytest.main([__file__])

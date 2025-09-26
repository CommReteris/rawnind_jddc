import pytest
from rawnind.inference.clean_api import InferenceConfig

class TestInferenceConfig:
    """Unit tests for InferenceConfig dataclass and its validation."""

    def test_valid_denoiser_config_init(self):
        """Test happy path initialization for a denoiser config."""
        config = InferenceConfig(
            architecture="unet",
            input_channels=3,
            device="cpu",
            filter_units=48,
            match_gain="output",
            enable_preupsampling=False,
            metrics_to_compute=["psnr", "ms_ssim"],
            crop_size=192 # Valid for MS-SSIM
        )
        assert config.architecture == "unet"
        assert config.input_channels == 3
        assert config.device == "cpu"
        assert config.filter_units == 48
        assert config.match_gain == "output"
        assert config.enable_preupsampling == False
        assert config.metrics_to_compute == ["psnr", "ms_ssim"]
        assert config.crop_size == 192

    def test_valid_compressor_config_init(self):
        """Test happy path initialization for a compressor config."""
        config = InferenceConfig(
            architecture="ManyPriors",
            input_channels=3,
            device="cuda:0",
            encoder_arch="Balle",
            decoder_arch="Balle",
            hidden_out_channels=192,
            bitstream_out_channels=64,
        )
        assert config.architecture == "ManyPriors"
        assert config.input_channels == 3
        assert config.device == "cuda:0"
        assert config.encoder_arch == "Balle"
        assert config.decoder_arch == "Balle"
        assert config.hidden_out_channels == 192
        assert config.bitstream_out_channels == 64
        
    @pytest.mark.parametrize(
        "param, value, expected_match",
        [
            ("architecture", "unsupported_arch", "Unsupported architecture"),
            ("input_channels", 0, "Input channels must be 3 or 4"),
            ("input_channels", 5, "Input channels must be 3 or 4"),
            ("match_gain", "invalid_gain", "Invalid match_gain option"),
        ],
    )
    def test_invalid_inference_config_raises_error(self, param, value, expected_match):
        """Test that invalid configurations raise ValueError."""
        base_config_params = {
            "architecture": "unet",
            "input_channels": 3,
            "device": "cpu",
        }
        base_config_params[param] = value
        with pytest.raises(ValueError, match=expected_match):
            InferenceConfig(**base_config_params)

    def test_preupsampling_validation(self):
        """Test enable_preupsampling constraint."""
        # Valid: Bayer (4ch) with preupsampling
        config_valid = InferenceConfig(
            architecture="unet", input_channels=4, device="cpu", enable_preupsampling=True
        )
        assert config_valid.enable_preupsampling == True

        # Invalid: RGB (3ch) with preupsampling
        with pytest.raises(ValueError, match="Preupsampling can only be used with 4-channel"):
            InferenceConfig(
                architecture="unet", input_channels=3, device="cpu", enable_preupsampling=True
            )

    @pytest.mark.parametrize("arch", ["unet", "utnet3", "identity", "bm3d", "ManyPriors", "DenoiseThenCompress", "JPEGXL", "JPEG", "Passthrough", "standard", "autoencoder"])
    def test_all_supported_architectures(self, arch):
        """Test that all specified supported architectures are valid."""
        config = InferenceConfig(architecture=arch, input_channels=3, device="cpu")
        assert config.architecture == arch # Should not raise ValueError
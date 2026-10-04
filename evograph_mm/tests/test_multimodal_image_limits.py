from PIL import Image
import pytest

from verl.utils.multimodal import process_image


def test_process_image_uses_environment_pixel_limits(monkeypatch):
    monkeypatch.setenv("EVOGRAPH_MM_MAX_IMAGE_PIXELS", "200704")
    monkeypatch.setenv("EVOGRAPH_MM_MIN_IMAGE_PIXELS", "50176")

    large = process_image(Image.new("RGB", (1600, 1200)))
    small = process_image(Image.new("RGB", (32, 32)))

    assert large.width * large.height <= 200704
    assert small.width * small.height >= 50176


def test_process_image_rejects_inverted_environment_limits(monkeypatch):
    monkeypatch.setenv("EVOGRAPH_MM_MAX_IMAGE_PIXELS", "100")
    monkeypatch.setenv("EVOGRAPH_MM_MIN_IMAGE_PIXELS", "101")

    with pytest.raises(ValueError, match="cannot exceed"):
        process_image(Image.new("RGB", (10, 10)))

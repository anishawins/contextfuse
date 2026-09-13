"""
Regression tests for the private-file filter.

The bug these exist to prevent: scanning a folder named "personal_ss"
matched the word "personal" in the ROOT folder name and skipped all 100
images in it, then reported "no readable images found".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.ingest import dhash, guess_source, hamming, looks_private  # noqa: E402


def test_root_folder_name_is_never_treated_as_private():
    """THE BUG. The user picked this folder deliberately."""
    root = Path("/Users/x/Downloads/personal_ss")
    assert not looks_private(root / "IMG_0536.PNG", root)
    assert not looks_private(Path("/Users/x/private_stuff/a.png"),
                             Path("/Users/x/private_stuff"))


def test_subfolder_below_root_is_still_checked():
    root = Path("/Users/x/shots")
    assert looks_private(root / "bank" / "a.png", root)
    assert not looks_private(root / "code" / "a.png", root)


def test_filename_is_always_checked():
    root = Path("/Users/x/shots")
    assert looks_private(root / "payslip_march.png", root)
    assert looks_private(root / "aadhaar.jpg", root)
    assert not looks_private(root / "Screenshot 2026-09-12.png", root)


def test_source_guessing():
    assert guess_source("Screenshot 2026-09-12 at 10.04.png") == "screenshot"
    assert guess_source("IMG_1234.HEIC") == "camera"
    assert guess_source("lecture_05_cnn.png") == "slides"
    assert guess_source("random.png") == "unknown"


def test_dhash_is_stable_and_discriminative():
    from PIL import Image
    a = Image.new("RGB", (200, 200), (10, 10, 10))
    for x in range(100):
        for y in range(200):
            a.putpixel((x, y), (240, 240, 240))
    b = a.resize((100, 100)).resize((200, 200))     # same image, requantised
    c = Image.new("RGB", (200, 200), (128, 0, 0))   # totally different

    assert hamming(dhash(a), dhash(b)) <= 6, "near-identical must hash close"
    assert hamming(dhash(a), dhash(c)) > 6, "different images must hash apart"

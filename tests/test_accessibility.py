"""
Accessibility audit using `axe-core` via Playwright.

Run as follows:
uv run pytest tests/test_accessibility.py -v --no-cov

Requires Streamlit weights to be present at:
checkpoints/default/cbis-whole-wd-only/best_model.safetensors

Some of the code in this script comes from axe-playwrite-python documentation.

Source: https://pamelafox.github.io/axe-playwright-python
"""

import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright
from axe_playwright_python.sync_playwright import Axe

APP_URL = "http://localhost:8502"
STARTUP_TIMEOUT = 15
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _wait_for_server(url: str, timeout: int) -> bool:
    """Wait for Streamlit server to start."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def _format_violations(results) -> str:
    count = results.violations_count
    if count == 0:
        return "No violations found."
    return results.generate_report()


@pytest.fixture(scope="module")
def streamlit_server():
    proc = subprocess.Popen(
        [
            "uv", "run", "streamlit", "run", "src/app.py",
            "--server.port=8502",
            "--server.headless=true",
            "--server.fileWatcherType=none",
        ],
        cwd=str(PROJECT_ROOT),
    )

    ready = _wait_for_server(APP_URL, STARTUP_TIMEOUT)
    if not ready:
        proc.terminate()
        pytest.fail(f"Streamlit did not start within {STARTUP_TIMEOUT}s")

    yield APP_URL

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def page(streamlit_server):
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(streamlit_server)
        pg.wait_for_selector(".stApp", timeout=10000)
        # Extra wait for Streamlit's to render.
        pg.wait_for_timeout(2000)
        yield pg
        browser.close()


def _click_tab(page, label: str):
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(1000)


def test_project_overview_tab(page):
    _click_tab(page, "Project Overview")
    results = Axe().run(page)
    report = _format_violations(results)
    print(report)
    assert results.violations_count == 0, report


def test_inference_tab(page):
    _click_tab(page, "Inference")
    results = Axe().run(page)
    report = _format_violations(results)
    print(report)
    assert results.violations_count == 0, report


def test_finetune_tab(page):
    _click_tab(page, "Fine-tune")
    results = Axe().run(page)
    report = _format_violations(results)
    print(report)
    assert results.violations_count == 0, report

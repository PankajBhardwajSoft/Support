"""Step 6: the saved transcripts still match what the scripts print (off by default: it runs 16 scripts)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(os.environ.get("OLA_CHECK_TRANSCRIPTS") != "1", reason="set OLA_CHECK_TRANSCRIPTS=1 to run")
def test_saved_transcripts_match_the_scripts():
    done = subprocess.run([sys.executable, os.path.join("scripts", "run_all_transcripts.py"), "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr

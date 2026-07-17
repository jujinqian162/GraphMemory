from __future__ import annotations

import os


# Prefect's ephemeral server may stop after pytest has closed its captured stream.
# Keep its asynchronous shutdown logs away from Rich's closed capture console.
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("PREFECT_LOGGING_INTERNAL_LEVEL", "ERROR")

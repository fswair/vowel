import importlib.util
import os
from typing import Any

import dotenv


def enable_monitoring(
    logfire_enabled: bool | Any = None,
    instrument_httpx: bool = False,
    httpx_capture_all: bool = True,
    **options,
):
    dotenv.load_dotenv()
    logfire_enabled = logfire_enabled or os.getenv("LOGFIRE_ENABLED")
    available = (
        logfire_enabled
        and isinstance(logfire_enabled, bool)
        or str(logfire_enabled).lower() == "true"
    )

    if available:
        if importlib.util.find_spec("logfire"):
            import logfire

            console = options.pop(
                "console",
                logfire.ConsoleOptions(show_project_link=False),
            )

            if "token" not in options and (token := os.getenv("LOGFIRE_TOKEN")):
                options["token"] = token

            logfire.configure(**options, console=console)
            logfire.instrument_pydantic_ai()

            if instrument_httpx:
                logfire.instrument_httpx(capture_all=httpx_capture_all)
        else:
            raise ImportError(
                "LOGFIRE_ENABLED is set but logfire is not installed. "
                "Please install logfire or set LOGFIRE_ENABLED=false"
            )

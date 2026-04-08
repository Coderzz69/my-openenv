"""FastAPI application for the AgriEnv OpenEnv server."""

from __future__ import annotations

import argparse
import os

from openenv.core.env_server.http_server import create_app

from agri_env.models import Action, Observation
from server.agri_environment import AgriEnvironment


# Enable the built-in OpenEnv web UI unless explicitly disabled.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")


app = create_app(
    AgriEnvironment,
    Action,
    Observation,
    env_name="agri_env",
    max_concurrent_envs=64,
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Run the AgriEnv server locally or in a container."""

    import uvicorn

    uvicorn.run(app, host=host, port=port or int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    os.environ["PORT"] = str(args.port)
    main()

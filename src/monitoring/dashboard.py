"""Monitoring dashboard for RetentionIQ.

Serves Evidently drift reports and basic model health status via a
lightweight FastAPI app. This is the backend for ``make monitor``.

The dashboard does NOT replace a full observability stack (Grafana, etc.).
It's a quick-look tool for the data science team to check drift status
and browse generated HTML reports.
"""

import argparse
import os
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

logger = structlog.get_logger()


def _load_config(config_path: str) -> dict:
    """Load monitoring config from YAML.

    Args:
        config_path: Path to the monitoring configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_monitoring_app(
    config_path: str = "configs/monitoring.yaml",
) -> FastAPI:
    """Create the monitoring dashboard FastAPI application.

    The app serves:
    - ``GET /`` — index page listing available reports
    - ``GET /health`` — health check
    - ``GET /reports`` — list generated report files
    - ``GET /reports/{filename}`` — serve a specific HTML report

    Args:
        config_path: Path to the monitoring YAML config.

    Returns:
        Configured FastAPI application instance.
    """
    config = _load_config(config_path)
    reports_dir = Path(
        config.get("reporting", {}).get("output_dir", "reports/monitoring/")
    )

    app = FastAPI(
        title="RetentionIQ Monitoring Dashboard",
        description="Drift detection and model performance reports",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict:
        """Health check for the monitoring dashboard."""
        return {
            "status": "healthy",
            "reports_dir": str(reports_dir),
            "reports_dir_exists": reports_dir.exists(),
        }

    @app.get("/reports")
    async def list_reports() -> dict:
        """List available monitoring reports."""
        if not reports_dir.exists():
            return {"reports": [], "count": 0}

        report_files = sorted(
            reports_dir.glob("*.html"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return {
            "reports": [
                {
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "url": f"/reports/{f.name}",
                }
                for f in report_files
            ],
            "count": len(report_files),
        }

    @app.get("/reports/{filename}")
    async def get_report(filename: str) -> FileResponse:
        """Serve a specific HTML report file.

        Args:
            filename: Name of the report file to serve.

        Returns:
            The HTML report file.

        Raises:
            HTTPException: If the file does not exist or is outside
                the reports directory.
        """
        # Prevent path traversal
        safe_name = Path(filename).name
        report_path = reports_dir / safe_name

        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report not found: {safe_name}",
            )

        return FileResponse(
            path=str(report_path),
            media_type="text/html",
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Render a simple index page listing reports."""
        if not reports_dir.exists():
            report_links = "<p>No reports directory found.</p>"
        else:
            report_files = sorted(
                reports_dir.glob("*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not report_files:
                report_links = "<p>No reports generated yet.</p>"
            else:
                items = "".join(
                    f'<li><a href="/reports/{f.name}">{f.name}</a>'
                    f" ({round(f.stat().st_size / 1024, 1)} KB)</li>"
                    for f in report_files
                )
                report_links = f"<ul>{items}</ul>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>RetentionIQ Monitoring</title></head>
        <body>
            <h1>RetentionIQ Monitoring Dashboard</h1>
            <h2>Available Reports</h2>
            {report_links}
            <hr>
            <p><a href="/health">Health Check</a>
             | <a href="/reports">Reports API</a></p>
        </body>
        </html>
        """

    logger.info(
        "monitoring_app_created",
        reports_dir=str(reports_dir),
    )

    return app


def main() -> None:
    """CLI entry point for ``make monitor``.

    Starts the monitoring dashboard server. Configuration is read
    from ``configs/monitoring.yaml`` by default, but can be overridden
    via the ``--config`` flag.
    """
    import uvicorn

    parser = argparse.ArgumentParser(
        description="RetentionIQ Monitoring Dashboard",
    )
    parser.add_argument(
        "--config",
        default="configs/monitoring.yaml",
        help="Path to monitoring config YAML",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MONITOR_PORT", "8050")),
        help="Port to bind to (default: 8050 or MONITOR_PORT env var)",
    )
    args = parser.parse_args()

    app = create_monitoring_app(config_path=args.config)

    logger.info(
        "starting_monitoring_dashboard",
        host=args.host,
        port=args.port,
        config=args.config,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

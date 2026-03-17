"""Bronze layer: raw data ingestion from parquet files.

Loads raw parquet data into DataFrames and validates schemas against
the definitions in configs/data.yaml. Dagster assets materialise each
table as a bronze-layer asset.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import structlog
import yaml
from dagster import AssetExecutionContext, asset

from src.exceptions import DataError, SchemaValidationError

logger = structlog.get_logger()

# Type-string to pandas dtype mapping used by schema validation.
_DTYPE_MAP: dict[str, str] = {
    "string": "object",
    "bool": "bool",
    "int64": "int64",
    "float64": "float64",
}


def load_raw_table(
    table_name: str,
    config_path: str = "configs/data.yaml",
) -> pd.DataFrame:
    """Load a raw parquet file for the given table.

    Args:
        table_name: Logical table name (e.g. ``"members"``).
        config_path: Path to the data pipeline config YAML.

    Returns:
        DataFrame with the raw contents of the parquet file.

    Raises:
        DataError: If the config is missing or the parquet file does
            not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise DataError(f"Config file not found: {config_path}")

    with open(config_file) as fh:
        config = yaml.safe_load(fh)

    raw_dir = Path(config["source"]["raw_data_dir"])
    table_cfg = _find_table_config(table_name, config)
    parquet_path = raw_dir / table_cfg["file"]

    if not parquet_path.exists():
        raise DataError(
            f"Parquet file not found for table '{table_name}': "
            f"{parquet_path}"
        )

    logger.info(
        "loading_raw_table",
        table=table_name,
        path=str(parquet_path),
    )
    df = pd.read_parquet(parquet_path)
    logger.info(
        "loaded_raw_table",
        table=table_name,
        rows=len(df),
        columns=list(df.columns),
    )
    return df


def validate_schema(
    df: pd.DataFrame,
    expected_schema: dict[str, str],
) -> bool:
    """Validate that *df* has the expected columns and compatible types.

    Args:
        df: DataFrame to validate.
        expected_schema: Mapping of column name to expected type
            string (as defined in ``configs/data.yaml``).

    Returns:
        ``True`` when validation passes.

    Raises:
        SchemaValidationError: When columns are missing or types do
            not match.
    """
    failures: list[str] = []
    expected_cols = set(expected_schema.keys())
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    if missing:
        failures.append(f"Missing columns: {sorted(missing)}")

    for col, expected_type in expected_schema.items():
        if col not in actual_cols:
            continue
        pandas_dtype = _DTYPE_MAP.get(expected_type, expected_type)
        actual_dtype = str(df[col].dtype)
        # Allow nullable boolean stored as object
        if pandas_dtype == "bool" and actual_dtype == "object":
            continue
        # Allow nullable int stored as float
        if pandas_dtype == "int64" and actual_dtype == "float64":
            continue
        if actual_dtype != pandas_dtype:
            failures.append(
                f"Column '{col}': expected {pandas_dtype}, "
                f"got {actual_dtype}"
            )

    if failures:
        raise SchemaValidationError(
            layer="bronze",
            suite="schema_check",
            failures=failures,
        )

    logger.info("schema_validation_passed", columns=len(expected_schema))
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_table_config(
    table_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the source-table entry for *table_name*."""
    for table in config["source"]["tables"]:
        if table["name"] == table_name:
            return table
    raise DataError(
        f"Table '{table_name}' not found in config. "
        f"Available: {[t['name'] for t in config['source']['tables']]}"
    )


def _load_config(
    config_path: str = "configs/data.yaml",
) -> dict[str, Any]:
    """Load and return the full data config dict."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _load_and_validate(
    table_name: str,
    config_path: str = "configs/data.yaml",
) -> pd.DataFrame:
    """Load a raw table and validate its schema in one step."""
    config = _load_config(config_path)
    df = load_raw_table(table_name, config_path)
    expected = config["bronze"]["schemas"][table_name]["columns"]
    validate_schema(df, expected)
    return df


# ---------------------------------------------------------------------------
# Dagster assets
# ---------------------------------------------------------------------------


@asset(
    group_name="bronze",
    description="Raw location data ingested from parquet.",
)
def bronze_locations(context: AssetExecutionContext) -> pd.DataFrame:
    """Ingest raw location data into the bronze layer."""
    df = _load_and_validate("locations")
    context.log.info(f"bronze_locations: {len(df)} rows loaded")
    return df


@asset(
    group_name="bronze",
    description="Raw member data ingested from parquet.",
)
def bronze_members(context: AssetExecutionContext) -> pd.DataFrame:
    """Ingest raw member data into the bronze layer."""
    df = _load_and_validate("members")
    context.log.info(f"bronze_members: {len(df)} rows loaded")
    return df


@asset(
    group_name="bronze",
    description="Raw visit data ingested from parquet.",
)
def bronze_visits(context: AssetExecutionContext) -> pd.DataFrame:
    """Ingest raw visit data into the bronze layer."""
    df = _load_and_validate("visits")
    context.log.info(f"bronze_visits: {len(df)} rows loaded")
    return df


@asset(
    group_name="bronze",
    description="Raw retention action data ingested from parquet.",
)
def bronze_retention_actions(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    """Ingest raw retention-action data into the bronze layer."""
    df = _load_and_validate("retention_actions")
    context.log.info(
        f"bronze_retention_actions: {len(df)} rows loaded"
    )
    return df

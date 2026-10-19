"""Loading the sampled tables and the fixed train/validation split."""

from __future__ import annotations

from typing import Any

import pandas as pd


# PROVIDED
def load_table(name: str, config: dict[str, Any]) -> pd.DataFrame:
    """Read one sampled table from data/sample/ with pandas' default type inference.

    Args:
        name: A table name from ``config["tables"]``, for example ``"bureau"``.
        config: The loaded config (see ``src.utils.load_config``).

    Returns:
        The table as a DataFrame.
    """
    if name not in config["tables"]:
        raise KeyError(f"Unknown table '{name}'. Known tables: {list(config['tables'])}")
    path = config["paths"]["sample_dir"] / config["tables"][name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run.py sample` first.")
    return pd.read_csv(path)


# PROVIDED
def load_application(config: dict[str, Any]) -> pd.DataFrame:
    """Return the sampled application_train table (one row per applicant)."""
    return load_table("application_train", config)


# TODO(student)
def load_split(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sampled application table into train and validation rows.

    Use the fixed split in ``config["paths"]["split_file"]`` (columns SK_ID_CURR and
    ``split``, where ``split`` is ``"train"`` or ``"valid"``). Never create your own split.

    Args:
        config: The loaded config.

    Returns:
        ``(train_df, valid_df)``: application rows, with all their original columns,
        for the train ids and the valid ids. Each has a fresh 0..n-1 index.

    Raises:
        ValueError: if the split file lists an id twice, uses a split value other than
            train/valid, or does not match the applicants in the sample one-to-one.

    Hints:
        - Read the id column so it has the same type as the id column in the application table.
        - Every applicant must land in exactly one of the two frames: no overlap, none missing.
        - Check the split file before you trust it, and fail loudly if it looks wrong.
        - Use ``config["id_col"]`` instead of typing the column name.
    """
    id_col = config["id_col"]
    app = load_application(config)
    split_path = config["paths"]["split_file"]
    split_df = pd.read_csv(split_path)
    required = {id_col, "split"}
    if not required.issubset(split_df.columns):
        raise ValueError(f"Split file must contain {sorted(required)}")
    split_df[id_col] = split_df[id_col].astype(app[id_col].dtype)
    if split_df[id_col].duplicated().any():
        raise ValueError("Split file contains duplicate applicant ids")
    if not split_df["split"].isin(["train", "valid"]).all():
        raise ValueError("Split file has values other than 'train' and 'valid'")
    app_ids = set(app[id_col].tolist())
    split_ids = set(split_df[id_col].tolist())
    if len(app_ids) != len(app) or app_ids != split_ids:
        raise ValueError("Split ids must match every sampled applicant exactly once")
    assignments = split_df.set_index(id_col)["split"]
    train = app.loc[app[id_col].map(assignments).eq("train")].reset_index(drop=True)
    valid = app.loc[app[id_col].map(assignments).eq("valid")].reset_index(drop=True)
    return train, valid

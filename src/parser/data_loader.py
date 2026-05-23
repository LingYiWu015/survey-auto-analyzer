"""Data Loader - loads survey response data from various formats.

Supports: CSV (various encodings), Excel (.xlsx/.xls), and auto-detection
of common survey platform export formats (问卷星, 腾讯问卷, Google Forms).
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Common encodings for Chinese survey data (try in order)
_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]


def detect_encoding(filepath: str, sample_size: int = 10000) -> str:
    """Try to detect file encoding by attempting to read with common encodings.

    Returns the first encoding that successfully decodes the file.
    """
    for enc in _ENCODINGS:
        try:
            with open(filepath, "r", encoding=enc) as f:
                f.read(sample_size)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"  # fallback


def load_csv(filepath: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load survey data from CSV file with auto encoding detection.

    Handles common Chinese survey export quirks:
    - Merged header rows (takes first non-empty row as header)
    - Trailing empty rows/columns
    - BOM markers
    """
    if encoding is None:
        encoding = detect_encoding(filepath)

    try:
        df = pd.read_csv(filepath, encoding=encoding, dtype=str)
    except Exception:
        # Try with different parameters for malformed CSVs
        df = pd.read_csv(
            filepath, encoding=encoding, dtype=str,
            on_bad_lines="skip", engine="python"
        )

    # Remove fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # If first row looks like a sub-header / question number row,
    # try to clean it up
    df = _clean_header_row(df)

    return df


def load_excel(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    """Load survey data from Excel file.

    Handles both .xlsx (openpyxl) and .xls (xlrd) formats.
    """
    engine = "openpyxl" if filepath.endswith(".xlsx") else "xlrd"
    df = pd.read_excel(filepath, sheet_name=sheet_name, dtype=str, engine=engine)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = _clean_header_row(df)
    return df


def _clean_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the header row.

    Some survey exports have the real question text on row 2 and
    question numbers on row 1. This detects and fixes that pattern.
    """
    if df.empty or len(df) < 1:
        return df

    # Check if first row values look like question numbers (e.g., "Q1", "1.", "第1题")
    first_row = df.iloc[0].astype(str)
    is_q_number_row = first_row.str.match(
        r"^(Q\d+|\d+\.?\s*|第\d+题).*"
    ).any()

    if is_q_number_row:
        # Check if second row has more descriptive text
        if len(df) > 1:
            second_row = df.iloc[1].astype(str)
            avg_len_1 = first_row.str.len().mean()
            avg_len_2 = second_row.str.len().mean()
            if avg_len_2 > avg_len_1 * 1.5:
                # Second row is likely the real header
                df = df.iloc[1:].reset_index(drop=True)

    # Clean column names: strip whitespace, truncate very long names
    df.columns = [str(c).strip()[:120] for c in df.columns]
    return df


def detect_platform_format(df: pd.DataFrame) -> str:
    """Detect which survey platform exported this data.

    Returns one of: 'wjx' (问卷星), 'tencent' (腾讯问卷), 'google', 'generic'
    """
    cols = " ".join(df.columns.astype(str)).lower()

    # 问卷星 indicators
    if any(kw in cols for kw in ["问卷星", "wjx", "提交时间", "ip地址", "序号"]):
        return "wjx"

    # 腾讯问卷 indicators
    if any(kw in cols for kw in ["腾讯问卷", "qq", "微信"]):
        return "tencent"

    # Google Forms indicators
    if "timestamp" in cols:
        return "google"

    return "generic"


def load_data(
    filepath: str, sheet_name: int = 0
) -> Tuple[pd.DataFrame, str]:
    """Main entry point: load survey data from any supported format.

    Returns (DataFrame, platform_name).
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = load_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = load_excel(filepath, sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .xlsx, .xls")

    platform = detect_platform_format(df)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {platform} format")

    return df, platform

import pandas as pd
import re
from typing import List


def load_data(path: str, sep: str = r"\t",) -> pd.DataFrame:
    """Loading the data from the specified path and returning it as pandas dataframe  for downstream processing.

    Args:
        path (str): Path to tsv / csv file to be loaded
        sep (str, optional): Separator used in the source data. Defaults to r'\t'.

    Returns:
        pd.DataFrame: Loaded dataframe
    """

    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = ["sequence_raw", "iRT"]

    df["sequence_length"] = df["sequence_raw"].apply(len)
    df["is_mod"] = df["sequence_raw"].str.contains("\[").astype(int)

    mod_index = df.query("is_mod == 1").index
    re_mod = re.compile(r"\[([\+A-Za-z0-9]+)\]")
    df.loc[mod_index, "modification"] = df.query("is_mod == 1")[
        "sequence_raw"
    ].str.findall(re_mod)

    return df


def replace_mod(s: str, modifications: List[str]) -> str:
    """_summary_

    Args:
        s (str): Input string
        modifications (List[str]): List of possible peptide modifications.

    Returns:
        str: List of all occurrences
    """

    for i, mod in enumerate(modifications):
        s = s.replace(mod, str(i + 1))

    return re.findall(r"[A-Z][0-9]?", s)

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Optional, Union

# local imports
from .data_ingestion import replace_mod


def preprocess_data(
    df: pd.DataFrame,
    vectorizer: Optional[Union[CountVectorizer, TfidfVectorizer]] = None,
    return_codes: bool = True,
) -> pd.DataFrame:
    """Data preprocessing and feature generation

    Args:
        df (pd.DataFrame): dataframe containing the raw data
        vectorizer (Union[CountVectorizer, TfidfVectorizer], optional): A sklearn vectorizer class turning the sequences into features. Defaults to None.
        return_codes (bool, optional): Return modifications as categorical data or numerical codes. Defaults to True.

    Returns:
        pd.DataFrame: preprocessed dataframe
    """

    # Get only rows with modified AAs
    mod_index = df.query("is_mod == 1").index
    re_mod = re.compile(r"\[([\+A-Za-z0-9]+)\]")
    df.loc[mod_index, "modification"] = df.query("is_mod == 1")[
        "sequence_raw"
    ].str.findall(re_mod)

    df.loc[mod_index, "modification_num"] = df.loc[mod_index, "modification"].apply(len)
    df.loc[mod_index, "modification_loc"] = df.query("is_mod == 1")[
        "sequence_raw"
    ].apply(lambda s: [match.span()[0] for match in re.finditer(re_mod, s)])

    # Get detailed information about the nature and position of the AA modifications
    max_mod = df.loc[mod_index, "modification_loc"].apply(len).max()
    for m in range(max_mod):
        df.loc[mod_index, f"modification_loc_{m + 1}"] = df.loc[
            mod_index, "modification_loc"
        ].apply(lambda l: l[m] if len(l) > m else -1)
        df.loc[mod_index, f"modification_type_{m + 1}"] = df.loc[
            mod_index, "modification"
        ].apply(lambda l: l[m] if len(l) > m else "")

        if return_codes:
            df.loc[:, f"modification_type_{m+1}"] = (
                df.loc[:, f"modification_type_{m+1}"]
                .fillna("")
                .astype("category")
                .cat.codes
            )
        else:
            df[f"modification_type_{m+1}"] = df[
                f"modification_type_{m+1}"
            ].cat.add_categories(
                -1
            )  # so we can use fillna!

    # Get Information about the sequence contents
    # Split the sequences
    mod_types = list(
        f"[{s}]"
        for s in df.query("is_mod == 1")["modification"].explode().unique()
        if len(s) > 0
    )
    df.loc[:, "sequence_proc"] = df.loc[:, "sequence_raw"].apply(
        lambda s: replace_mod(s, mod_types)
    )
    vocabulary = df["sequence_proc"].explode().unique()
    if vectorizer:

        vec = vectorizer(
            token_pattern=r"(?u)\b\w\d?\b", vocabulary=list(vocabulary), lowercase=False
        )
        df.loc[:, [f"AAcount_{v}" for v in vocabulary]] = vec.fit_transform(
            df["sequence_proc"].apply(lambda s: " ".join(s)).to_numpy()
        ).toarray()

    return df

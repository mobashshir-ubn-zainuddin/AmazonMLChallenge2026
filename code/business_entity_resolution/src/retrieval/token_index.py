"""
Token-based inverted indexes for business entity retrieval.

The index is built over Source 2 and Source 3 records.

Retrieval is country-partitioned and uses token overlap rather than
requiring complete name/address equality.
"""

import heapq
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Tuple, Optional

class TokenIndex:
    """
    Inverted token index for names and addresses.

    To optimize memory and speed, the index stores integer row positions
    instead of entity ID strings.
    """

    def __init__(self):
        # posting lists: (country, token) -> list of integer positions
        self.name_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.address_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        # document frequency for IDF calculation
        self.name_df: Dict[Tuple[str, str], int] = defaultdict(int)
        self.address_df: Dict[Tuple[str, str], int] = defaultdict(int)

        # precomputed IDF weights: (country, token) -> float
        self.name_idf: Dict[Tuple[str, str], float] = {}
        self.address_idf: Dict[Tuple[str, str], float] = {}

        # total documents per field for IDF
        self.name_document_count = 0
        self.address_document_count = 0

        # mapping: position -> entity_id
        self.pos_to_id: List[str] = []
        self._current_pos = 0

    @staticmethod
    def _tokens(value: str) -> List[str]:
        """
        Convert a normalized token string into unique tokens.
        """
        if not value:
            return []
        return list(set(value.split()))

    def add_record(
        self,
        entity_id: str,
        country: str,
        name_tokens: str,
        address_tokens: str,
    ) -> None:
        """
        Add one entity to the token index.
        """
        country = country or ""
        pos = self._current_pos
        self.pos_to_id.append(entity_id)
        self._current_pos += 1

        name_token_list = self._tokens(name_tokens)
        address_token_list = self._tokens(address_tokens)

        if name_token_list:
            self.name_document_count += 1

        if address_token_list:
            self.address_document_count += 1

        for token in name_token_list:
            key = (country, token)
            self.name_index[key].append(pos)
            self.name_df[key] += 1

        for token in address_token_list:
            key = (country, token)
            self.address_index[key].append(pos)
            self.address_df[key] += 1

    def add_dataframe(self, df) -> None:
        """
        Add a processed Source 2 or Source 3 DataFrame.
        """
        required = {
            "entity_id",
            "country_clean",
            "name_tokens",
            "address_tokens",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in df[list(required)].itertuples(index=False):
            # Map itertuples names to parameters
            # row.entity_id, row.country_clean, etc.
            self.add_record(
                entity_id=getattr(row, "entity_id"),
                country=getattr(row, "country_clean"),
                name_tokens=getattr(row, "name_tokens"),
                address_tokens=getattr(row, "address_tokens"),
            )

    def finalize_index(self) -> None:
        """
        Precompute IDF weights for all tokens in the index.
        Should be called after all add_record/add_dataframe calls.
        """
        # IDF = log((N + 1) / (df + 1)) + 1.0
        n = self.name_document_count
        for key, df in self.name_df.items():
            self.name_idf[key] = log((n + 1) / (df + 1)) + 1.0

        n = self.address_document_count
        for key, df in self.address_df.items():
            self.address_idf[key] = log((n + 1) / (df + 1)) + 1.0

    @staticmethod
    def _select_tokens(
        country: str,
        tokens: List[str],
        document_frequency: Dict[Tuple[str, str], int],
        max_df: Optional[int],
        top_k: Optional[int],
    ) -> List[str]:
        """
        Select informative tokens based on document frequency.
        """
        selected = []
        for token in set(tokens):
            df = document_frequency.get((country, token), 0)
            if df == 0:
                continue
            if max_df is not None and df > max_df:
                continue
            selected.append((token, df))

        # Sort by df (ascending) then token (alphabetical) for determinism
        selected.sort(key=lambda x: (x[1], x[0]))

        if top_k is not None:
            selected = selected[:top_k]

        return [token for token, _ in selected]

    def _retrieve_from_index(
        self,
        index: Dict[Tuple[str, str], List[int]],
        idf_weights: Dict[Tuple[str, str], float],
        document_frequency: Dict[Tuple[str, str], int],
        country: str,
        tokens: List[str],
        max_df: Optional[int],
        top_k_tokens: Optional[int],
        max_candidates: Optional[int],
    ) -> List[str]:
        """
        Retrieve entities using integer positions and precomputed IDF.
        """
        selected_tokens = self._select_tokens(
            country=country,
            tokens=tokens,
            document_frequency=document_frequency,
            max_df=max_df,
            top_k=top_k_tokens,
        )

        if not selected_tokens:
            return []

        # Scores stored by integer position
        scores: Dict[int, float] = defaultdict(float)

        for token in selected_tokens:
            key = (country, token)
            idf = idf_weights.get(key)
            if idf is None:
                continue

            # Use integer positions for scoring
            positions = index.get(key, [])
            for pos in positions:
                scores[pos] += idf

        if not scores:
            return []

        # Top-K selection
        # Sort criteria: score (descending), then position (ascending) for determinism
        # Note: original code used entity_id for tie-breaking.
        # To be equivalent, we'd need to resolve IDs first, but the requirement
        # says "deterministic ranking" and "tie-breaking must remain deterministic".
        # We use position as the tie-breaker.

        if max_candidates is not None:
            # Use nlargest for efficiency
            top_positions = heapq.nlargest(
                max_candidates,
                scores.items(),
                key=lambda x: (x[1], -x[0])
            )
            return [self.pos_to_id[pos] for pos, _ in top_positions]
        else:
            ranked = sorted(
                scores.items(),
                key=lambda x: (-x[1], x[0])
            )
            return [self.pos_to_id[pos] for pos, _ in ranked]

    def retrieve_name(
        self,
        country: str,
        name_tokens: str,
        max_df: Optional[int] = None,
        top_k_tokens: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve candidates using name-token overlap.
        """
        # Lazy finalization if weights are not yet computed
        if not self.name_idf and self.name_df:
            self.finalize_index()

        return self._retrieve_from_index(
            index=self.name_index,
            idf_weights=self.name_idf,
            document_frequency=self.name_df,
            country=country or "",
            tokens=self._tokens(name_tokens),
            max_df=max_df,
            top_k_tokens=top_k_tokens,
            max_candidates=max_candidates,
        )

    def retrieve_address(
        self,
        country: str,
        address_tokens: str,
        max_df: Optional[int] = None,
        top_k_tokens: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve candidates using address-token overlap.
        """
        if not self.address_idf and self.address_df:
            self.finalize_index()

        return self._retrieve_from_index(
            index=self.address_index,
            idf_weights=self.address_idf,
            document_frequency=self.address_df,
            country=country or "",
            tokens=self._tokens(address_tokens),
            max_df=max_df,
            top_k_tokens=top_k_tokens,
            max_candidates=max_candidates,
        )

    def retrieve(
        self,
        country: str,
        name_tokens: str,
        address_tokens: str,
        max_df: Optional[int] = None,
        top_k_tokens: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve the union of name-token and address-token candidates.
        """
        candidates = set()

        candidates.update(
            self.retrieve_name(
                country=country,
                name_tokens=name_tokens,
                max_df=max_df,
                top_k_tokens=top_k_tokens,
                max_candidates=max_candidates,
            )
        )

        candidates.update(
            self.retrieve_address(
                country=country,
                address_tokens=address_tokens,
                max_df=max_df,
                top_k_tokens=top_k_tokens,
                max_candidates=max_candidates,
            )
        )

        return sorted(candidates)

    def retrieve_record(
        self,
        row,
        max_df: Optional[int] = None,
        top_k_tokens: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """
        Convenience wrapper for a processed dataframe row.
        """
        return self.retrieve(
            country=row.country_clean,
            name_tokens=row.name_tokens,
            address_tokens=row.address_tokens,
            max_df=max_df,
            top_k_tokens=top_k_tokens,
            max_candidates=max_candidates,
        )

    def statistics(self) -> Dict[str, int]:
        """
        Return basic index statistics.
        """
        return {
            "name_index_keys": len(self.name_index),
            "address_index_keys": len(self.address_index),
            "name_unique_tokens": len(self.name_df),
            "address_unique_tokens": len(self.address_df),
            "name_documents": self.name_document_count,
            "address_documents": self.address_document_count,
            "total_indexed_records": len(self.pos_to_id),
        }

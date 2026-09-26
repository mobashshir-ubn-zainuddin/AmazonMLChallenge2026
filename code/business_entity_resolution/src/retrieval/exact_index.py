"""
Exact retrieval indexes for business entity resolution.

Indexes are built over noisy Source 2 and Source 3 records and queried
using Source 1 records.

All indexes are country-partitioned so that records are only retrieved
within the same normalized country.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


class ExactIndex:
    """
    Collection of exact-match inverted indexes.

    Each index maps a normalized key to the entity IDs sharing that key.
    """

    def __init__(self):
        self.name_exact: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.name_no_suffix_exact: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.address_exact: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.name_address_exact: Dict[
            Tuple[str, str, str], List[str]
        ] = defaultdict(list)

    def add_record(
        self,
        entity_id: str,
        country: str,
        name_clean: str,
        name_no_suffix: str,
        address_clean: str,
    ) -> None:
        """
        Add one Source 2/Source 3 record to the exact indexes.
        """

        country = country or ""
        name_clean = name_clean or ""
        name_no_suffix = name_no_suffix or ""
        address_clean = address_clean or ""

        if name_clean:
            self.name_exact[(country, name_clean)].append(entity_id)

        if name_no_suffix:
            self.name_no_suffix_exact[
                (country, name_no_suffix)
            ].append(entity_id)

        if address_clean:
            self.address_exact[
                (country, address_clean)
            ].append(entity_id)

        if name_no_suffix and address_clean:
            self.name_address_exact[
                (country, name_no_suffix, address_clean)
            ].append(entity_id)

    def add_dataframe(self, df) -> None:
        """
        Add a processed Source 2 or Source 3 DataFrame.

        Required columns:
            entity_id
            country_clean
            name_clean
            name_no_legal_suffix
            address_clean
        """

        required = {
            "entity_id",
            "country_clean",
            "name_clean",
            "name_no_legal_suffix",
            "address_clean",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        for row in df[
            [
                "entity_id",
                "country_clean",
                "name_clean",
                "name_no_legal_suffix",
                "address_clean",
            ]
        ].itertuples(index=False):
            self.add_record(
                entity_id=row.entity_id,
                country=row.country_clean,
                name_clean=row.name_clean,
                name_no_suffix=row.name_no_legal_suffix,
                address_clean=row.address_clean,
            )

    @staticmethod
    def _lookup(
        index: Dict,
        key: Tuple,
    ) -> List[str]:
        """
        Return matching entity IDs for an exact key.
        """

        if any(value == "" for value in key):
            return []

        return index.get(key, [])

    def retrieve(
        self,
        country: str,
        name_clean: str,
        name_no_suffix: str,
        address_clean: str,
    ) -> List[str]:
        """
        Retrieve the union of all exact-index matches for one Source 1 record.
        """

        country = country or ""
        name_clean = name_clean or ""
        name_no_suffix = name_no_suffix or ""
        address_clean = address_clean or ""

        candidates = set()

        candidates.update(
            self._lookup(
                self.name_exact,
                (country, name_clean),
            )
        )

        candidates.update(
            self._lookup(
                self.name_no_suffix_exact,
                (country, name_no_suffix),
            )
        )

        candidates.update(
            self._lookup(
                self.address_exact,
                (country, address_clean),
            )
        )

        candidates.update(
            self._lookup(
                self.name_address_exact,
                (
                    country,
                    name_no_suffix,
                    address_clean,
                ),
            )
        )

        return sorted(candidates)

    def retrieve_record(self, row) -> List[str]:
        """
        Convenience wrapper for a pandas row / namedtuple-like object.
        """

        return self.retrieve(
            country=row.country_clean,
            name_clean=row.name_clean,
            name_no_suffix=row.name_no_legal_suffix,
            address_clean=row.address_clean,
        )

    def statistics(self) -> Dict[str, int]:
        """
        Return basic index statistics.
        """

        return {
            "name_exact_keys": len(self.name_exact),
            "name_no_suffix_exact_keys": len(
                self.name_no_suffix_exact
            ),
            "address_exact_keys": len(self.address_exact),
            "name_address_exact_keys": len(
                self.name_address_exact
            ),
        }
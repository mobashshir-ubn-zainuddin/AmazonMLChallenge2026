import sys
import os
sys.path.append(os.path.join(os.getcwd(), "code"))
from business_entity_resolution.src.retrieval.token_index import TokenIndex
import pandas as pd

def test_token_index():
    print("Running tiny correctness test...")
    index = TokenIndex()
    
    # Sample data
    # S2: 2 records
    s2_df = pd.DataFrame({
        "entity_id": ["S2_1", "S2_2"],
        "country_clean": ["USA", "USA"],
        "name_tokens": ["amazon web services", "amazon logistics"],
        "address_tokens": ["seattle washington", "new york york"],
    })
    # S3: 1 record (Different country)
    s3_df = pd.DataFrame({
        "entity_id": ["S3_1"],
        "country_clean": ["CAN"],
        "name_tokens": ["amazon canada"],
        "address_tokens": ["toronto ontario"],
    })
    
    index.add_dataframe(s2_df)
    index.add_dataframe(s3_df)
    
    # 1. Test Country Partitioning (S1 in USA should not see S3 in CAN)
    res_usa = index.retrieve_name("USA", "amazon", max_df=100, top_k_tokens=5, max_candidates=10)
    assert "S3_1" not in res_usa, "Country partitioning failed: USA query saw CAN record"
    assert "S2_1" in res_usa and "S2_2" in res_usa, "USA query missed USA records"
    print("OK: Country partitioning")

    # 2. Test Name Token Retrieval
    res_name = index.retrieve_name("USA", "services", max_df=100, top_k_tokens=5, max_candidates=10)
    assert res_name == ["S2_1"], f"Name retrieval failed: expected ['S2_1'], got {res_name}"
    print("OK: Name token retrieval")

    # 3. Test Address Token Retrieval
    res_addr = index.retrieve_address("USA", "york", max_df=100, top_k_tokens=5, max_candidates=10)
    assert res_addr == ["S2_2"], f"Address retrieval failed: expected ['S2_2'], got {res_addr}"
    print("OK: Address token retrieval")

    # 4. Test max_candidates
    # Query "amazon" in USA -> should find S2_1 and S2_2. Limit to 1.
    res_limit = index.retrieve_name("USA", "amazon", max_df=100, top_k_tokens=5, max_candidates=1)
    assert len(res_limit) == 1, f"max_candidates failed: expected length 1, got {len(res_limit)}"
    print("OK: max_candidates")

    # 5. Test empty/missing fields
    res_empty = index.retrieve_name("USA", "", max_df=100, top_k_tokens=5, max_candidates=10)
    assert res_empty == [], f"Empty query should return nothing, got {res_empty}"
    print("OK: Empty fields")

    print("\nAll tiny tests passed!")

if __name__ == "__main__":
    test_token_index()

"""Hidden tests for w3-merge.

Hardened: 40 customers + 120 seeded orders with unmatched keys on both sides,
duplicate orders per customer, independent row-count verification via set
intersection, name-id correspondence check, left/outer join traps.
"""
import numpy as np
import pandas as pd
from solution import join_orders


def _make_data(seed=7):
    rng = np.random.default_rng(seed)
    # 40 customers with IDs 1..40
    cust_ids = np.arange(1, 41)
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "name": [f"customer_{i:03d}" for i in cust_ids],
    })
    # 120 orders: customer_ids drawn from 1..50 → 41..50 have NO matching customer
    order_cids = rng.integers(1, 51, size=120)
    amounts = rng.uniform(5.0, 999.0, size=120).round(2)
    # Shuffle so order is not correlated with customer_id
    perm = rng.permutation(120)
    orders = pd.DataFrame({
        "customer_id": order_cids[perm],
        "amt": amounts[perm],
    })
    return customers, orders


def test_inner_join_basic():
    """Original small regression test."""
    c = pd.DataFrame({"customer_id": [1, 2], "name": ["a", "b"]})
    o = pd.DataFrame({"customer_id": [1, 1, 3], "amt": [10, 20, 30]})
    out = join_orders(c, o)
    assert len(out) == 2
    assert set(out.columns) == {"customer_id", "name", "amt"}
    assert out["amt"].tolist() == [10, 20]


def test_inner_join_row_count():
    """Row count must equal orders whose customer_id appears in customers."""
    customers, orders = _make_data()
    out = join_orders(customers, orders)
    cust_id_set = set(customers["customer_id"])
    expected_count = int(orders["customer_id"].isin(cust_id_set).sum())
    assert len(out) == expected_count, (
        f"Expected {expected_count} rows (inner join), got {len(out)}"
    )


def test_inner_join_no_unmatched_orders():
    """Orders with no matching customer (IDs 41–50) must be absent from output."""
    customers, orders = _make_data()
    out = join_orders(customers, orders)
    cust_id_set = set(customers["customer_id"])
    assert out["customer_id"].isin(cust_id_set).all(), (
        "Output contains customer_ids that are not in customers table"
    )


def test_inner_join_no_childless_customers():
    """Customers with no orders must not appear in the inner-join output."""
    customers, orders = _make_data()
    out = join_orders(customers, orders)
    order_cid_set = set(orders["customer_id"])
    # Any customer_id in output must also appear in orders
    assert out["customer_id"].isin(order_cid_set).all()


def test_inner_join_columns():
    """Output must contain exactly customer_id, name, amt."""
    customers, orders = _make_data()
    out = join_orders(customers, orders)
    assert set(out.columns) == {"customer_id", "name", "amt"}


def test_inner_join_name_correspondence():
    """Every row's name must match its customer_id per the customers table."""
    customers, orders = _make_data()
    out = join_orders(customers, orders)
    name_map = dict(zip(customers["customer_id"], customers["name"]))
    mismatches = [(r["customer_id"], r["name"]) for _, r in out.iterrows()
                  if r["name"] != name_map[r["customer_id"]]]
    assert mismatches == [], f"Name-ID mismatches: {mismatches}"


def test_inner_join_duplicate_orders():
    """Customer with multiple orders yields multiple rows (not collapsed)."""
    # Give customer 5 exactly 3 orders
    customers = pd.DataFrame({"customer_id": [5, 6, 7], "name": ["c5", "c6", "c7"]})
    orders = pd.DataFrame({
        "customer_id": [5, 5, 5, 6, 9],  # 9 has no matching customer
        "amt": [100.0, 200.0, 300.0, 50.0, 999.0],
    })
    out = join_orders(customers, orders)
    assert len(out) == 4  # 3 orders for c5 + 1 for c6; customer 9 dropped
    assert (out["customer_id"] == 5).sum() == 3
    assert (out["customer_id"] == 9).sum() == 0


def test_inner_join_all_unmatched():
    """If no orders match any customer, result must be empty (not error)."""
    customers = pd.DataFrame({"customer_id": [1, 2], "name": ["a", "b"]})
    orders = pd.DataFrame({"customer_id": [99, 100], "amt": [10.0, 20.0]})
    out = join_orders(customers, orders)
    assert len(out) == 0
    assert set(out.columns) == {"customer_id", "name", "amt"}

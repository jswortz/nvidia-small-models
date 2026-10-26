"""Build the synthetic retail store used by the POC.

Creates data/store.db (SQLite) and data/policy.md. Output is deterministic,
so rerunning this script is also how you reset the store before an eval run.

Usage:
    python data/generate.py
    python data/generate.py --db /tmp/store.db --policy /tmp/policy.md
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TODAY = date(2026, 10, 26)  # fixed "today" so expected answers never drift
SEED = 42

# (product_id, name, category, price, final_sale)
PRODUCTS = [
    ("P001", "AeroBrew Coffee Maker", "kitchen", 89.99, 0),
    ("P002", "AeroBrew Espresso Machine", "kitchen", 349.00, 0),
    ("P003", "CrispAir Air Fryer 5qt", "kitchen", 119.99, 0),
    ("P004", "CrispAir Air Fryer 8qt", "kitchen", 159.99, 0),
    ("P005", "SteadyHeat Toaster 2-Slice", "kitchen", 39.99, 0),
    ("P006", "SteadyHeat Toaster 4-Slice", "kitchen", 59.99, 0),
    ("P007", "PrimeCut Knife Set", "kitchen", 129.00, 0),
    ("P008", "ChefLine Stand Mixer", "kitchen", 279.00, 0),
    ("P009", "ChefLine Hand Mixer", "kitchen", 49.99, 0),
    ("P010", "FreshKeep Food Storage Set", "kitchen", 34.99, 1),
    ("P011", "VortexPro Blender 600W", "kitchen", 69.99, 0),
    ("P012", "VortexPro Blender 900W", "kitchen", 99.99, 0),
    ("P013", "VortexPro Blender 1200W", "kitchen", 149.99, 0),
    ("P014", "PureFlow Water Filter Pitcher", "kitchen", 29.99, 0),
    ("P015", "CloudRest Pillow 2-Pack", "home", 59.99, 0),
    ("P016", "CloudRest Weighted Blanket", "home", 119.00, 0),
    ("P017", "BrightNest Smart Bulb 4-Pack", "home", 49.99, 0),
    ("P018", "BrightNest Smart Plug 2-Pack", "home", 24.99, 1),
    ("P019", "QuietBreeze Tower Fan", "home", 89.99, 0),
    ("P020", "QuietBreeze Air Purifier", "home", 229.00, 0),
    ("P021", "TidyBot Robot Vacuum", "home", 399.00, 0),
    ("P022", "TidyBot Robot Vacuum Pro", "home", 649.00, 0),
    ("P023", "ClearView 55in 4K TV", "electronics", 549.00, 0),
    ("P024", "ClearView 65in 4K TV", "electronics", 1299.00, 0),
    ("P025", "SoundArc Soundbar", "electronics", 199.00, 0),
    ("P026", "SoundArc Wireless Earbuds", "electronics", 129.00, 0),
    ("P027", "PulseFit Smartwatch", "electronics", 249.00, 0),
    ("P028", "VoltGo Power Bank 20000mAh", "electronics", 39.99, 0),
    ("P029", "VoltGo USB-C Charger 65W", "electronics", 34.99, 1),
    ("P030", "StreamStick 4K", "electronics", 49.99, 0),
]

FIRST = ["Avery", "Jordan", "Riley", "Morgan", "Casey", "Quinn", "Harper", "Rowan",
         "Emerson", "Finley", "Sage", "Reese", "Parker", "Dakota", "Hayden"]
LAST = ["Nguyen", "Patel", "Garcia", "Kim", "Okafor", "Silva", "Novak", "Larsen",
        "Haddad", "Rossi", "Murphy", "Tanaka", "Moreau", "Kowalski", "Reyes"]

# Orders that eval cases depend on. Keep these stable.
PINNED = {
    "1187": dict(product_id="P012", status="delivered", delivered_days_ago=32),   # cracked blender on day 32
    "1042": dict(product_id="P024", status="delivered", delivered_days_ago=10),   # $1,299 TV, refund needs approval
    "1100": dict(product_id="P019", status="delivered", delivered_days_ago=45),   # outside the return window
    "1150": dict(product_id="P018", status="delivered", delivered_days_ago=5),    # final sale item
    "1199": dict(product_id="P003", status="processing", delivered_days_ago=None),  # not shipped yet
    "1023": dict(product_id="P026", status="refunded", delivered_days_ago=20),    # already refunded
}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL, final_sale INTEGER
);
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, email TEXT);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT, product_id TEXT, quantity INTEGER,
    total REAL, order_date TEXT, delivered_date TEXT, status TEXT
);
CREATE TABLE refunds (
    refund_id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, amount REAL,
    reason TEXT, created_at TEXT
);
CREATE TABLE replacements (
    replacement_id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, new_product_id TEXT,
    price_difference REAL, reason TEXT, created_at TEXT
);
"""

POLICY = """# Returns, Refunds, and Warranty Policy

## 1. Returns

### 1.1 Return window
Most items can be returned within 30 days of the delivery date for a full refund. The 30 day return window starts on the day the order is delivered.

### 1.2 Condition of returned items
Returned items must include all original parts and accessories. Opened items are accepted if they are undamaged and complete.

### 1.3 Final sale items
Items marked final sale, including clearance products, cannot be returned or refunded unless they arrive damaged or defective.

### 1.4 Orders not yet shipped
An order that is still processing can be cancelled for a full refund at any time before it ships.

## 2. Refunds

### 2.1 Refund method and timing
Approved refunds go back to the original payment method within 5 to 7 business days.

### 2.2 Shipping costs
Original shipping costs are not refundable, except when the item arrived damaged, broken, cracked, or defective.

## 3. Exchanges

### 3.1 Exchanges within the return window
Within 30 days of delivery, an item can be exchanged for a different size, color, or model in the same category. If the new item costs more, the customer pays the price difference. If it costs less, we refund the difference.

## 4. Damaged or defective items

### 4.1 Damage reported within 7 days
If an item arrives damaged, broken, or cracked and the customer reports it within 7 days of delivery, they can choose a full refund or a free replacement.

### 4.2 Replacement for damaged or defective items within 60 days
Items that are damaged, broken, cracked, or defective can be replaced within 60 days of delivery, even after the 30 day return window. The replacement can be the same product or a different model in the same category. If the different model costs more, the customer pays the price difference. If it costs less, we refund the difference.

### 4.3 Photos of damage
For damage claims after 7 days, ask the customer for a photo of the damage before issuing the replacement.

## 5. Warranty

### 5.1 Manufacturer warranty
Appliances and electronics carry a one year manufacturer warranty. After 60 days from delivery, defects are handled through the manufacturer warranty, not through store replacement.

## 6. Escalation

### 6.1 High value refunds
Any refund over $500 requires approval from a human supervisor before it is issued.

### 6.2 Requests outside this policy
Requests this policy does not allow, such as a refund after the return window for an item that is not damaged, must be politely declined or handed to a human agent. Do not make exceptions.
"""


def build(db_path: Path, policy_path: Path) -> None:
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO meta VALUES ('today', ?)", (TODAY.isoformat(),))
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)

    customers = []
    for i in range(1, 61):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        customers.append((f"C{i:03d}", f"{first} {last}", f"{first}.{last}{i}@example.com".lower()))
    con.executemany("INSERT INTO customers VALUES (?,?,?)", customers)

    prices = {p[0]: p[3] for p in PRODUCTS}
    orders, refunds = [], []
    for n in range(1001, 1201):
        oid = str(n)
        cust = rng.choice(customers)[0]
        pid = rng.choice(PRODUCTS)[0]
        qty = 1 if rng.random() < 0.85 else 2
        ordered = TODAY - timedelta(days=rng.randint(1, 120))
        delivered = None
        if (TODAY - ordered).days <= 2:
            status = "processing"
        else:
            arrival = ordered + timedelta(days=rng.randint(3, 6))
            if arrival > TODAY:
                status = "shipped"
            else:
                delivered, status = arrival, "delivered"
                if rng.random() < 0.08:
                    status = "refunded"

        if oid in PINNED:
            pin = PINNED[oid]
            pid, qty, status = pin["product_id"], 1, pin["status"]
            if pin["delivered_days_ago"] is None:
                delivered, ordered = None, TODAY - timedelta(days=1)
            else:
                delivered = TODAY - timedelta(days=pin["delivered_days_ago"])
                ordered = delivered - timedelta(days=4)

        total = round(prices[pid] * qty, 2)
        orders.append((oid, cust, pid, qty, total, ordered.isoformat(),
                       delivered.isoformat() if delivered else None, status))
        if status == "refunded":
            refunded_on = min(delivered + timedelta(days=3), TODAY)
            refunds.append((oid, total, "customer return", refunded_on.isoformat()))

    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)
    con.executemany(
        "INSERT INTO refunds (order_id, amount, reason, created_at) VALUES (?,?,?,?)", refunds
    )
    con.commit()
    con.close()

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(POLICY)
    print(f"Wrote {db_path} ({len(PRODUCTS)} products, {len(customers)} customers, "
          f"{len(orders)} orders, {len(refunds)} refunds) and {policy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the synthetic store")
    parser.add_argument("--db", type=Path, default=HERE / "store.db")
    parser.add_argument("--policy", type=Path, default=HERE / "policy.md")
    args = parser.parse_args()
    build(args.db, args.policy)

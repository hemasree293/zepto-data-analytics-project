from pathlib import Path
import sqlite3
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORY_URLS = {
    "Romance": BASE_URL + "catalogue/category/books/romance_8/index.html",
    "Mystery": BASE_URL + "catalogue/category/books/mystery_3/index.html",
    "Horror": BASE_URL + "catalogue/category/books/horror_31/index.html",
}
FIXED_GBP_TO_INR = 105.50
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
DB_PATH = DB_DIR / "books.db"
RAW_PATH = DATA_DIR / "books_raw.csv"
CLEAN_PATH = DATA_DIR / "books_cleaned.csv"
QUERIES_PATH = ROOT / "sql" / "queries.sql"
QUERY_OUTPUT_PATH = OUTPUT_DIR / "query_outputs.txt"


def scrape_category(session, category_name, start_url):
    rows = []
    url = start_url
    while url:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        products = soup.select("article.product_pod")
        if not products:
            break

        for product in products:
            title_tag = product.select_one("h3 a")
            price_tag = product.select_one(".price_color")
            rating_tag = product.select_one(".star-rating")
            availability_tag = product.select_one(".availability")

            rows.append({
                "title": title_tag.get("title", title_tag.get_text(strip=True)) if title_tag else None,
                "price": price_tag.get_text(strip=True) if price_tag else None,
                "star_rating": " ".join(rating_tag.get("class", []))[1:] if rating_tag else None,
                "availability": availability_tag.get_text(" ", strip=True) if availability_tag else None,
                "category": category_name,
            })

        next_link = soup.select_one("li.next a")
        if next_link and next_link.get("href"):
            next_url = requests.compat.urljoin(url, next_link["href"])
            url = next_url
            time.sleep(0.15)
        else:
            url = None

    return rows


def scrape_all():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Data Pipeline Assignment)"})
    all_rows = []
    for category, url in CATEGORY_URLS.items():
        all_rows.extend(scrape_category(session, category, url))
    raw = pd.DataFrame(all_rows)
    if len(raw) < 60 or raw["category"].nunique() < 3:
        raise ValueError("Scrape did not meet the minimum of 60 books across 3 categories.")
    raw.to_csv(RAW_PATH, index=False)
    return raw


def clean_data(raw):
    df = raw.copy()
    df["price_gbp"] = pd.to_numeric(
        df["price"].astype("string").str.replace("£", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    rating_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    df["rating"] = df["star_rating"].astype("string").str.strip().str.lower().map(rating_map)

    availability = df["availability"].astype("string").str.strip().str.lower()
    df["in_stock"] = availability.map(
        lambda x: True if x == "in stock" else (False if "out of stock" in x else pd.NA)
    )

    # Numeric parsing failures are median-imputed as required by the assignment.
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median()).round()

    # Availability is categorical/Boolean, so guessing is unsafe; drop only those rows.
    df = df.dropna(subset=["in_stock"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    df["price_inr"] = df["price_gbp"] * FIXED_GBP_TO_INR
    df.to_csv(CLEAN_PATH, index=False)
    return df


def create_database(df):
    DB_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER CHECK (rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    categories = pd.DataFrame({"category_name": sorted(df["category"].unique())})
    categories.to_sql("categories", conn, if_exists="append", index=False)
    category_lookup = pd.read_sql("SELECT category_id, category_name FROM categories", conn)

    books = df.merge(category_lookup, left_on="category", right_on="category_name", how="left")
    books = books[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].copy()
    books["in_stock"] = books["in_stock"].astype(int)
    books.to_sql("books", conn, if_exists="append", index=False)
    conn.close()


def run_queries():
    queries = {
        "Q1_SELECT_WHERE": """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC, price_gbp DESC;
""".strip(),
        "Q2_ORDER_BY_LIMIT": """
SELECT title, price_gbp, price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
""".strip(),
        "Q3_DISTINCT": """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
""".strip(),
        "Q4_IN": """
SELECT title, rating, category_id
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, title
LIMIT 15;
""".strip(),
        "Q5_BETWEEN": """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
""".strip(),
        "Q6_JOIN_TOP_RATED_PER_CATEGORY": """
WITH ranked AS (
    SELECT
        b.title,
        b.price_gbp,
        b.rating,
        b.in_stock,
        c.category_name,
        ROW_NUMBER() OVER (
            PARTITION BY c.category_id
            ORDER BY b.rating DESC, b.price_gbp DESC, b.title
        ) AS rn
    FROM books AS b
    JOIN categories AS c ON b.category_id = c.category_id
)
SELECT category_name, title, price_gbp, rating, in_stock
FROM ranked
WHERE rn <= 10
ORDER BY category_name, rating DESC, price_gbp DESC, title;
""".strip(),
    }

    QUERIES_PATH.write_text("\n\n".join(f"-- {name}\n{sql}" for name, sql in queries.items()), encoding="utf-8")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    with QUERY_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for name, sql in queries.items():
            result = pd.read_sql(sql, conn)
            f.write(f"\n{'=' * 80}\n{name}\n{'=' * 80}\n")
            f.write(result.to_string(index=False))
            f.write("\n")
    conn.close()
    return queries


def pandas_validation(df, queries):
    conn = sqlite3.connect(DB_PATH)
    sql_join = pd.read_sql(queries["Q6_JOIN_TOP_RATED_PER_CATEGORY"], conn)
    conn.close()

    # Reproduce the JOIN in pandas without SQL.
    categories = pd.read_sql("SELECT * FROM categories", sqlite3.connect(DB_PATH))
    books = pd.read_sql("SELECT * FROM books", sqlite3.connect(DB_PATH))
    merged = books.merge(categories, on="category_id", how="inner")
    merged["in_stock"] = merged["in_stock"].astype(bool)
    merged["rn"] = (
        merged.sort_values(["category_id", "rating", "price_gbp", "title"], ascending=[True, False, False, True])
        .groupby("category_id")
        .cumcount() + 1
    )
    pandas_join = merged.loc[merged["rn"] <= 10, ["category_name", "title", "price_gbp", "rating", "in_stock"]]
    pandas_join = pandas_join.sort_values(["category_name", "rating", "price_gbp", "title"], ascending=[True, False, False, True]).reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)

    # Normalize numeric/Boolean representations for comparison.
    sql_join["in_stock"] = sql_join["in_stock"].astype(bool)
    equivalent = sql_join.equals(pandas_join)

    sql_join.to_csv(OUTPUT_DIR / "join_pd_read_sql.csv", index=False)
    pandas_join.to_csv(OUTPUT_DIR / "join_pd_merge.csv", index=False)
    pd.concat(
        [sql_join.add_prefix("sql_"), pandas_join.add_prefix("pandas_")], axis=1
    ).to_csv(OUTPUT_DIR / "join_side_by_side.csv", index=False)
    (OUTPUT_DIR / "join_equivalence.txt").write_text(
        f"pd.read_sql JOIN result equals pd.merge result: {equivalent}\n",
        encoding="utf-8",
    )
    return equivalent


def main():
    for directory in [DATA_DIR, DB_DIR, OUTPUT_DIR, LOG_DIR, ROOT / "sql"]:
        directory.mkdir(exist_ok=True)

    raw = scrape_all()
    clean = clean_data(raw)
    create_database(clean)
    queries = run_queries()
    equivalent = pandas_validation(clean, queries)

    summary = [
        f"Scraped rows: {len(raw)}",
        f"Categories: {raw['category'].nunique()}",
        f"Cleaned rows: {len(clean)}",
        f"Fixed GBP->INR rate: {FIXED_GBP_TO_INR}",
        f"JOIN SQL vs pandas.merge equivalent: {equivalent}",
        f"SQLite database: {DB_PATH}",
    ]
    (LOG_DIR / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

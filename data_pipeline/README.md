# Module 1 — Data Pipeline

## Objective

This module implements a complete catalog-style data pipeline:

**scrape → clean → fixed-rate currency conversion → normalized SQLite → SQL queries → pandas validation**

The source is [Books to Scrape](https://books.toscrape.com/), a public scraping-practice website.

## Dataset scope

The scraper collects all books from three categories:

- Romance
- Mystery
- Horror

The current dataset contains **84 books across 3 categories** (35 Romance, 32 Mystery, 17 Horror), satisfying the requirement of at least 60 books across at least 3 categories.

## Required fixed currency rate

> **1 GBP = 105.50 INR**

This is an artificial, project-defined baseline rate for this assignment. It is not a live or historical market exchange rate. No external currency API is used.

`price_inr` is calculated as:

```text
price_inr = price_gbp * 105.50
```

## Cleaning decisions

- `price` is stripped of the `£` symbol and converted to numeric `price_gbp`.
- `star_rating` values `One`–`Five` are mapped to integer `rating` values `1`–`5`.
- `availability` is converted to Boolean `in_stock` (`True` for `In stock`, `False` for `Out of stock`).
- Numeric parsing failures are converted to missing values and median-imputed, so unexpected numeric text does not crash the pipeline.
- An unrecognized availability value is not safely inferable as in-stock or out-of-stock, so that row is dropped rather than guessed.
- The database stores Boolean values using SQLite's integer convention: `1=True`, `0=False`.

## Database design

### `categories`

| Column | Type | Constraint |
|---|---|---|
| `category_id` | INTEGER | Primary key |
| `category_name` | TEXT | NOT NULL, UNIQUE |

### `books`

| Column | Type | Constraint |
|---|---|---|
| `book_id` | INTEGER | Primary key |
| `title` | TEXT | NOT NULL |
| `price_gbp` | REAL | — |
| `price_inr` | REAL | — |
| `rating` | INTEGER | CHECK 1–5 |
| `in_stock` | INTEGER | CHECK 0/1 |
| `category_id` | INTEGER | Foreign key → `categories.category_id` |

The relationship is one-to-many: one category can contain many books.

## Installation

From the `data_pipeline` directory:

```bash
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python run_pipeline.py
```

The script automatically:

1. Scrapes the three categories using `requests` and `BeautifulSoup`.
2. Saves raw scraped data to `data/books_raw.csv`.
3. Cleans and converts the fields.
4. Calculates `price_inr` using the fixed rate above.
5. Creates `database/books.db` from scratch.
6. Inserts categories and books using a PK/FK relationship.
7. Executes six SQL queries.
8. Saves SQL output to `output/query_outputs.txt`.
9. Reads the JOIN result using `pd.read_sql()`.
10. Reproduces the JOIN using `pd.merge()` and checks that both outputs are equivalent.

## SQL requirements covered

`sql/queries.sql` contains six queries covering:

- `SELECT` / `WHERE`
- `ORDER BY`
- `LIMIT`
- `DISTINCT`
- `IN`
- `BETWEEN`
- `JOIN`
- Window function `ROW_NUMBER()` for the top-rated books within each category

## Pandas validation

The following files document the two approaches to the JOIN:

- `output/join_pd_read_sql.csv` — JOIN result read from SQLite with `pd.read_sql()`.
- `output/join_pd_merge.csv` — equivalent JOIN reproduced with `pd.merge()`.
- `output/join_side_by_side.csv` — both results placed side by side.
- `output/join_equivalence.txt` — equality check.

## Repository structure

```text
data_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── data/
│   └── books_cleaned.csv
├── database/
│   └── books.db
├── sql/
│   └── queries.sql
├── output/
│   ├── query_outputs.txt
│   ├── join_pd_read_sql.csv
│   ├── join_pd_merge.csv
│   ├── join_side_by_side.csv
│   └── join_equivalence.txt
└── logs/
    └── run_summary.txt
```

## Note about the included CSV

`data/books_cleaned.csv` is included as the cleaned dataset used to validate the database/query artifacts in this submission. The authoritative reproduction path is `run_pipeline.py`, which performs the live scrape from Books to Scrape and rebuilds all downstream outputs from scratch.
## Currency Conversion

The project uses the fixed baseline conversion rate:

**1 GBP = 105.50 INR**

This is an artificial, project-defined rate used consistently for this assignment. No live currency API is used.
## Module 1 Design Decisions

- **Scraping:** `requests` and `BeautifulSoup` are used to collect catalog data from the specified book categories.
- **Cleaning:** Price, rating, and availability fields are converted into consistent numeric/Boolean representations.
- **Currency conversion:** A fixed project-defined rate of `1 GBP = 105.50 INR` is used so results are deterministic and reproducible.
- **Database:** SQLite is used because it is lightweight, local, and requires no external database server.
- **Normalization:** Categories and books are stored in separate tables using a primary-key/foreign-key relationship.
- **Validation:** The SQL JOIN result is reproduced with `pandas.merge()` and compared for equivalence.

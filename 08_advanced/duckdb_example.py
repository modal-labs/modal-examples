import modal

stub = modal.Stub(image=modal.DebianSlim().pip_install(["duckdb", "pandas", "yfinance"]))
shared_volume = modal.SharedVolume().persist("db_storage")


ALL_TICKERS = ["MSFT", "AAPL", "GOOG", "NFLX", "AMZN"]


def get_db():
    import duckdb
    return duckdb.connect("/db/duck.db")


@stub.function(shared_volumes={"/db": shared_volume})
def bootstrap():
    import yfinance
    ticker_data = yfinance.download(" ".join(ALL_TICKERS))
    bootstrap_df = ticker_data["Adj Close"].melt(ignore_index=False)
    bootstrap_df["date"] = bootstrap_df.index
    db = get_db()
    db.execute("""CREATE OR REPLACE TABLE stock_prices AS (
        SELECT
            date,
            variable AS ticker,
            value AS price
        FROM bootstrap_df
        WHERE value IS NOT NULL
    )""")


@stub.function(shared_volumes={"/db": shared_volume})
def print_stats():
    db = get_db()
    stats = db.execute("""
        SELECT
            ticker,
            date,
            price,
            AVG(price) OVER (
                PARTITION BY ticker
                ORDER BY date
                ROWS BETWEEN 30 PRECEDING AND 0 FOLLOWING
            ) as rolling_30d
        FROM stock_prices""")
    print(stats.df())


if __name__ == "__main__":
    with stub.run():
        bootstrap()
        print_stats()

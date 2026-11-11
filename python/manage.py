"""Run schema migration and optional explicit demo seed before API startup."""
import argparse
from config import get_settings
from services.repository import Repository

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action='store_true')
    args = parser.parse_args()
    repo = Repository(get_settings().database_url)
    try:
        repo.migrate()
        if args.seed:
            from services.seed_data import SEED_PRODUCTS
            repo.seed(SEED_PRODUCTS)
    finally:
        repo.engine.dispose()

if __name__ == '__main__':
    main()

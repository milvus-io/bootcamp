import argparse
from service.api import app


def app_runner():
    app.run(host="0.0.0.0", debug=True, port=5000)


def run_with_args():
    parser = argparse.ArgumentParser(description='Start args')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    if args.debug:
        app_runner()


if __name__ == "__main__":
    run_with_args()

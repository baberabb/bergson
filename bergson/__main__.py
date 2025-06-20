from simple_parsing import parse

from .build import build_index
from .data import IndexConfig


def main():
    build_index(parse(IndexConfig))


if __name__ == "__main__":
    main()

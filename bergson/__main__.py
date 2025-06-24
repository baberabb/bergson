from simple_parsing import parse

from .build import build_gradient_dataset
from .data import IndexConfig


def main():
    build_gradient_dataset(parse(IndexConfig))


if __name__ == "__main__":
    main()

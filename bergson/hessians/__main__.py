from simple_parsing import parse

from bergson.data import IndexConfig
from bergson.hessians.compute_EKFAC import compute_EKFAC


def main():
    compute_EKFAC(parse(IndexConfig))


if __name__ == "__main__":
    main()

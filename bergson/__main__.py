from simple_parsing import parse

from bergson.distributed import distributed_computing

from .collection import collect_gradients
from .data import IndexConfig


def main():
    distributed_computing(
        parse(IndexConfig),
        worker_fn=collect_gradients,
    )


if __name__ == "__main__":
    main()

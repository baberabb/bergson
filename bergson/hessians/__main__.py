from simple_parsing import parse

from bergson.data import IndexConfig
from bergson.distributed import distributed_computing
from bergson.hessians.compute_all import compute_all_factors


def main():
    distributed_computing(cfg=parse(IndexConfig), worker_fn=compute_all_factors)


if __name__ == "__main__":
    main()

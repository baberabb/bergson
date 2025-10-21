import os
from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build_gradient_dataset
from .data import IndexConfig, QueryConfig
from .dynamic_query import query_gradient_dataset
from .static_query import query_existing


@dataclass
class StaticQuery:
    """Query an on-disk gradient index."""

    query_cfg: QueryConfig

    index_cfg: IndexConfig

    k: int | None = None

    def execute(self):
        """Query an on-disk gradient index."""
        assert self.query_cfg.scores_path
        assert self.query_cfg.query_path

        query_existing(self.query_cfg, self.index_cfg, self.k)


@dataclass
class Build:
    """Build the gradient dataset."""

    cfg: IndexConfig

    def execute(self):
        """Build the gradient dataset."""
        if not self.cfg.save_index and not self.cfg.save_processor:
            raise ValueError(
                "At least one of save_index or save_processor must be True"
            )

        build_gradient_dataset(self.cfg)


@dataclass
class Query:
    """Query the gradient dataset."""

    query_cfg: QueryConfig

    index_cfg: IndexConfig

    def execute(self):
        """Query the gradient dataset."""
        assert self.query_cfg.scores_path
        assert self.query_cfg.query_path

        if os.path.exists(self.index_cfg.run_path) and self.index_cfg.save_index:
            raise ValueError(
                "Index path already exists and save_index is True - "
                "running this query will overwrite the existing gradients. "
                "If you meant to query the existing gradients use "
                "Attributor instead."
            )

        query_gradient_dataset(self.query_cfg, self.index_cfg)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Build, Query, StaticQuery]

    def execute(self):
        """Run the script."""
        self.command.execute()


def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()

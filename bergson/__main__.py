import os
from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build
from .data import IndexConfig, QueryConfig
from .query import query


@dataclass
class Build:
    """Build the gradient dataset."""

    cfg: IndexConfig

    def execute(self):
        """Build the gradient dataset."""
        if not self.cfg.save_index and self.cfg.skip_preconditioners:
            raise ValueError(
                "Either save_index must be True or skip_preconditioners must be False"
            )

        build(self.cfg)


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

        query(self.index_cfg, self.query_cfg)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Build, Query]

    def execute(self):
        """Run the script."""
        self.command.execute()


def get_parser():
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None):
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()

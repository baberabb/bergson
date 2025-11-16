import os
from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build
from .data import IndexConfig, ReduceConfig, ScoreConfig
from .query.query_index import QueryConfig, query
from .reduce import reduce
from .score.score import score_dataset


@dataclass
class Build:
    """Build a gradient index."""

    cfg: IndexConfig

    def execute(self):
        """Build the gradient index."""
        if not self.cfg.save_index and self.cfg.skip_preconditioners:
            raise ValueError(
                "Either save_index must be True or skip_preconditioners must be False"
            )

        build(self.cfg)


@dataclass
class Reduce:
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    reduce_cfg: ReduceConfig

    def execute(self):
        """Reduce a gradient index."""
        reduce(self.index_cfg, self.reduce_cfg)


@dataclass
class Score:
    """Score a dataset against an existing gradient index."""

    score_cfg: ScoreConfig

    index_cfg: IndexConfig

    def execute(self):
        """Score a dataset against an existing gradient index."""
        assert self.score_cfg.scores_path
        assert self.score_cfg.query_path

        if os.path.exists(self.index_cfg.run_path) and self.index_cfg.save_index:
            raise ValueError(
                "Index path already exists and save_index is True - "
                "running this will overwrite the existing gradients. "
                "If you meant to query an existing gradient dataset use "
                "Attributor instead."
            )

        score_dataset(self.index_cfg, self.score_cfg)


@dataclass
class Query:
    """Query an existing gradient index."""

    cfg: QueryConfig

    def execute(self):
        """Query an existing gradient index."""
        query(self.cfg)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Build, Query, Reduce, Score]

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

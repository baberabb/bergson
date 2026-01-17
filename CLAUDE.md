Always test your changes by running the appropriate script or CLI command.

## Project Structure and Conventions

Keep __main__.py clean - it should primarily provide documentation and routing for the available CLI commands and their configs.

Consider writing a new library file if you add a standalone, complex feature used in more than one place.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

Never save logs, scripts, and other random development into the root of a project. Create an appropriate directory such as runs/ or scripts/ and add it to the .gitignore.

# Development

You can call CLI commands without prefixing `python -m`, like `bergson build`.

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

Don't remove large datasets from the HF cache without asking.

### Tests

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

### Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate && pip install pytest`.

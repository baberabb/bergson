Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

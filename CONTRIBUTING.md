# Contributing

Contributions to the environment, methods, examples and documentation are welcome.
For a new method, start with the [baseline guide](docs/baselines.md) and
[`Controller` interface](src/hedging_gym/interfaces.py).

## Add or improve a method

1. Create a branch in your clone or fork. Keep the change focused on one method
   or one environment improvement.
2. Add a named module in `src/hedging_gym/baselines/`. Its header should link the
   paper, upstream source and implementation notes, and identify any changes to
   the published method. Preserve upstream licenses when porting code.
3. Use the environment's observations, accounting and evaluation. A controller
   returns target holdings; it does not modify the ledger. Reuse established
   learning libraries where appropriate. Training can remain method-specific.
4. Add a runnable example or benchmark command and focused checks for the new
   behavior. Update the method catalogue and source mapping.
5. Open a pull request explaining the method, how to run it and what you checked.
   Maintainers squash integration commits while retaining contributor credit.

## Check a change

```bash
uv sync --locked
uv run --frozen pytest tests/test_methods.py
uv run --frozen python -m hedging_gym.validate --device cpu
```

Choose additional tests for the code you changed. Use
`uv sync --locked --group baselines` for SB3-based methods. External author
runtimes have separate instructions in the method guides.

## Report research results

Use the [reproducibility guide](docs/baseline-implementation.md). Keep the
configuration and evaluator fixed when comparing methods; report all declared
training seeds, budgets and relevant source differences. Store checkpoints,
raw paths and logs outside Git. Publish reviewed results with the commands and
artifacts needed to reproduce them, rather than an isolated winning score.

Do not add internal meeting notes, approval logs, machine-specific paths or
temporary experiment summaries to the documentation. Explain behavior and
scientific limitations in terms a user can act on.

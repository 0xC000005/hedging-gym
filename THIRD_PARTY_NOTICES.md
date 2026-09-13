# Third-party notices

Original Hedging Gym code and documentation use the [MIT License](LICENSE).
We acknowledge the following research software and retain its license notices.
Method descriptions, paper citations and implementation changes are documented
in the [baseline guide](docs/baseline-methods.md) and source headers.

| Component | Upstream source | Retained license |
|---|---|---|
| Belief-context encoder and prediction loss | [BeliefConditionedFB](https://github.com/maxsbob/BeliefConditionedFB/tree/30e7487ca033c3619ec744ed55f916ece005c425), modified PyTorch port | [Apache-2.0](docs/belief-fb-LICENSE.txt) |
| Shared QR/EX-D4PG learner | [Cao/Hull learner](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/blob/77dc48326da000d983b1fb750edb2177e38c75fd/agent/learning.py) and [EX-DRL learner](https://github.com/pmalekzadeh/EX-DRL/blob/f1abe99df7fa9efaa65af6b9dd416c3425c64098/agent/learning.py), adapted to PyTorch and the common hedging objective | [Apache-2.0](docs/belief-fb-LICENSE.txt); Copyright 2018 DeepMind Technologies Limited. All rights reserved. |
| EX-DRL | [Author repository](https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098) | [MIT](LICENSES/EX-DRL-MIT.txt); Copyright (c) 2022 rotmanfinhub. Individual Apache-licensed files retain their notices. |
| Hybrid policy optimization | [hybrid-rl](https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659), mixed-gradient method adapted to hedging | [MIT](LICENSES/HPO-MIT.txt), as supplied upstream |

Hull DDPG and common AlphaZero are local reimplementations informed by the cited
papers and reference code. QR-D4PG uses a local quantile critic and the shared
learner credited above. Method headers describe their implementation differences;
this does not relicense the authors' reference repositories.

Separately installed dependencies and author checkouts retain their own terms.
External donor code loaded by wrappers such as `source_alphazero.py` is not
included in this package. Complete retained license texts accompany both source
and wheel distributions.

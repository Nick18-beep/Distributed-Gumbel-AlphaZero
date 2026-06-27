from __future__ import annotations

import torch

from gumbel_az.config import load_config
from gumbel_az.envs.custom.connect_four import COLUMNS, ConnectFourGame
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.masking import masked_policy
from gumbel_az.search.torch_gumbel_backend import TorchGumbelSearchBackend

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _network_apply(observation):
    batch_size = observation.shape[0]
    logits = torch.arange(COLUMNS, dtype=torch.float32, device=observation.device)
    return NetworkOutput(
        policy_logits=logits[None, :].repeat(batch_size, 1),
        value=torch.zeros((batch_size,), dtype=torch.float32, device=observation.device),
    )


def test_torch_gumbel_search_masks_illegal_actions_and_is_deterministic() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = game.init()
    observations = torch.as_tensor(game.canonical_observation(state)[None, ...]).repeat(2, 1, 1, 1)
    legal_mask = torch.as_tensor(game.legal_action_mask(state)[None, ...]).repeat(2, 1)
    legal_mask[:, -1] = False
    backend = TorchGumbelSearchBackend(game=game)
    generator_a = torch.Generator().manual_seed(0)
    generator_b = torch.Generator().manual_seed(0)

    output_a = backend.search(
        root_observation=observations,
        root_legal_mask=legal_mask,
        network_apply=_network_apply,
        rng=generator_a,
        config=config.search,
        root_embedding=[state, state],
    )
    output_b = backend.search(
        root_observation=observations,
        root_legal_mask=legal_mask,
        network_apply=_network_apply,
        rng=generator_b,
        config=config.search,
        root_embedding=[state, state],
    )

    assert output_a.policy_target.shape == (2, COLUMNS)
    assert output_a.visit_counts.shape == (2, COLUMNS)
    assert output_a.q_values.shape == (2, COLUMNS)
    assert output_a.prior_logits.shape == (2, COLUMNS)
    assert output_a.policy_target[:, -1].tolist() == [0.0, 0.0]
    assert torch.allclose(output_a.policy_target.sum(dim=-1), torch.ones(2))
    assert torch.equal(output_a.selected_action, output_b.selected_action)
    assert torch.allclose(output_a.policy_target, output_b.policy_target)


def test_masked_policy_assigns_zero_to_illegal_actions() -> None:
    logits = torch.asarray([[1.0, 2.0, 3.0]])
    legal = torch.asarray([[True, False, True]])
    probs = masked_policy(logits, legal)

    assert probs[0, 1] == 0.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_torch_gumbel_search_rejects_shape_mismatch() -> None:
    config = load_config(CONFIG)
    backend = TorchGumbelSearchBackend()

    try:
        backend.search(
            root_observation=torch.zeros((1, 6, 7, 2), dtype=torch.float32),
            root_legal_mask=torch.ones((1, COLUMNS + 1), dtype=torch.bool),
            network_apply=_network_apply,
            rng=torch.Generator().manual_seed(0),
            config=config.search,
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_torch_gumbel_search_rejects_roots_without_legal_actions() -> None:
    config = load_config(CONFIG)
    backend = TorchGumbelSearchBackend()

    try:
        backend.search(
            root_observation=torch.zeros((1, 6, 7, 2), dtype=torch.float32),
            root_legal_mask=torch.zeros((1, COLUMNS), dtype=torch.bool),
            network_apply=_network_apply,
            rng=torch.Generator().manual_seed(0),
            config=config.search,
        )
    except ValueError as exc:
        assert "legal action" in str(exc)
    else:
        raise AssertionError("expected all-illegal root to fail")


def test_torch_gumbel_search_visit_counts_do_not_mark_illegal_padding() -> None:
    config = load_config(CONFIG, ["search.max_num_considered_actions=7"])
    backend = TorchGumbelSearchBackend()
    legal_mask = torch.asarray([[False, False, True, False, False, False, False]])

    output = backend.search(
        root_observation=torch.zeros((1, 6, 7, 2), dtype=torch.float32),
        root_legal_mask=legal_mask,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(0),
        config=config.search,
    )

    assert output.selected_action.tolist() == [2]
    assert output.policy_target[0, 2] == 1.0
    assert torch.count_nonzero(output.policy_target[~legal_mask]) == 0
    assert torch.count_nonzero(output.visit_counts[~legal_mask]) == 0


def test_torch_gumbel_search_rejects_root_embedding_batch_mismatch() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = game.init()
    observations = torch.as_tensor(game.canonical_observation(state)[None, ...]).repeat(2, 1, 1, 1)
    legal_mask = torch.as_tensor(game.legal_action_mask(state)[None, ...]).repeat(2, 1)
    backend = TorchGumbelSearchBackend(game=game)

    try:
        backend.search(
            root_observation=observations,
            root_legal_mask=legal_mask,
            network_apply=_network_apply,
            rng=torch.Generator().manual_seed(0),
            config=config.search,
            root_embedding=state,
        )
    except ValueError as exc:
        assert "root_embedding batch size" in str(exc)
    else:
        raise AssertionError("expected root embedding batch mismatch to fail")


def test_torch_gumbel_search_accepts_tuple_root_embedding_batch() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = game.init()
    observations = torch.as_tensor(game.canonical_observation(state)[None, ...]).repeat(2, 1, 1, 1)
    legal_mask = torch.as_tensor(game.legal_action_mask(state)[None, ...]).repeat(2, 1)
    backend = TorchGumbelSearchBackend(game=game)

    output = backend.search(
        root_observation=observations,
        root_legal_mask=legal_mask,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(0),
        config=config.search,
        root_embedding=(state, state),
    )

    assert output.policy_target.shape == (2, COLUMNS)


def test_torch_gumbel_search_uses_terminal_child_reward_for_q_value() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = game.init()
    for action in [0, 1, 0, 1, 0, 1]:
        state = game.step(state, action)
    legal_mask = torch.as_tensor(game.legal_action_mask(state)[None, ...])
    observation = torch.as_tensor(game.canonical_observation(state)[None, ...])
    backend = TorchGumbelSearchBackend(game=game)

    output = backend.search(
        root_observation=observation,
        root_legal_mask=legal_mask,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(0),
        config=config.search,
        root_embedding=[state],
    )

    assert output.q_values[0, 0] == 1.0

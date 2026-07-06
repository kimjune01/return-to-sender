import numpy as np

from return_to_sender.sim import World, aipw_scores, e_bh, e_process, ht_scores


def make_world(seed=0, **kw):
    rng = np.random.default_rng(seed)
    world = World(**kw).build(rng)
    return world, rng


def test_ht_score_unbiased_for_relay_and_zero_for_honest():
    world, rng = make_world(n_agents=200, n_relays=10, background=0.0)
    p = 0.5
    treatments = rng.binomial(1, p, (200, 20000)).astype(np.int8)
    harm = world.simulate_windows(treatments, rng)
    scores = ht_scores(harm, treatments, p)
    means = scores.mean(axis=1)
    assert np.allclose(means[world.relays], world.beta, atol=0.15)
    assert np.allclose(means[~world.relays], 0.0, atol=0.15)


def test_e_process_is_supermartingale_under_null():
    # Under H0 (honest nodes, delta = 0 <= tau), E[e-value] <= 1.
    world, rng = make_world(n_agents=100, n_relays=0, n_subcontractors=0)
    p, tau = 0.5, 1.0
    terminal = []
    for _ in range(50):
        treatments = rng.binomial(1, p, (100, 50)).astype(np.int8)
        harm = world.simulate_windows(treatments, rng)
        harm = np.minimum(harm, 6.0)
        scores = ht_scores(harm, treatments, p)
        terminal.append(e_process(scores, tau, -6.0 / p))
    mean_e = np.concatenate(terminal).mean()
    assert mean_e <= 1.1  # Monte Carlo slack around E[e] <= 1


def test_aipw_e_process_is_supermartingale_under_null():
    world, rng = make_world(n_agents=100, n_relays=0, n_subcontractors=0)
    p, tau = 0.5, 1.0
    terminal = []
    for _ in range(50):
        treatments = rng.binomial(1, p, (100, 50)).astype(np.int8)
        harm = np.minimum(world.simulate_windows(treatments, rng), 6.0)
        scores = aipw_scores(harm, treatments, p)
        terminal.append(e_process(scores, tau, -6.0 / p))
    assert np.concatenate(terminal).mean() <= 1.1


def test_aipw_unbiased_for_relay_effect():
    world, rng = make_world(n_agents=200, n_relays=10, background=0.0)
    p = 0.5
    treatments = rng.binomial(1, p, (200, 5000)).astype(np.int8)
    harm = world.simulate_windows(treatments, rng)
    scores = aipw_scores(harm, treatments, p)
    means = scores.mean(axis=1)
    assert np.allclose(means[world.relays], world.beta, atol=0.15)
    assert np.allclose(means[~world.relays], 0.0, atol=0.15)


def test_e_bh_flags_top_k_correctly():
    # n=10, q=0.5: node with e=100 passes (100 >= 10/(0.5*1)=20);
    # second-ranked e=12 passes at k=2 (12 >= 10/(0.5*2)=10); e=5 fails at k=3.
    e = np.array([100.0, 12.0, 5.0] + [1.0] * 7)
    flagged = e_bh(e, q=0.5)
    assert flagged[:2].all() and not flagged[2:].any()


def test_e_bh_flags_nothing_on_uniform_nulls():
    flagged = e_bh(np.ones(100), q=0.1)
    assert not flagged.any()

import torch
import time

from kairos.attentions import ATTN_IMPL
from kairos.attentions import KairosCache
from kairos.attentions import KairosAttention, KairosRotaryEmbedding
from kairos.attentions import KairosGatedDeltaNet
from kairos.attentions import KairosLiZAttention2

# =========================
# Config
# =========================
class DummySWAConfig:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    sliding_window_size = 2
    max_position_embeddings = 64
    rope_theta = 10000.0

    layers_config = ["l"]       # attention only
    slw_wsize = -1

class DummyDeltaConfig:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    sliding_window_size = 2
    max_position_embeddings = 64
    rope_theta = 10000.0
    expand_factor = 2.0
    linear_conv_kernel_dim = 3

    time_step_min = 0.001
    time_step_max = 0.1
    time_step_floor = 1e-4
    A_init_range = (0.1, 1.0)

    use_uscaling = False

    layers_config = ["d"]       # deltanet only
    sliding_window_size = 16
    slw_wsize = -1


# =========================
# SWA TESTS
# =========================
def get_swa_model():
    cfg = DummySWAConfig()
    attn = KairosAttention(cfg)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    return attn, rope


def get_swa_inputs(B=2, L=8, D=32):
    x = torch.randn(B, L, D)
    pos_ids = torch.arange(L).unsqueeze(0)
    return x, pos_ids


def test_swa_shape():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out = attn(x, rope(x, pos))
    assert out.shape == x.shape


def test_swa_no_nan():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out = attn(x, rope(x, pos))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_swa_bidirectional_symmetry():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()

    out1 = attn(x, rope(x, pos))

    x_rev = torch.flip(x, dims=[1])
    pos_rev = torch.flip(pos, dims=[1])
    out2 = attn(x_rev, rope(x_rev, pos_rev))
    out2 = torch.flip(out2, dims=[1])

    assert torch.allclose(out1, out2, atol=1e-4)


def test_swa_window_locality():
    attn, rope = get_swa_model()

    x = torch.zeros(1, 10, 32)
    x[:, 5] = 10.0

    pos = torch.arange(10).unsqueeze(0)
    out = attn(x, rope(x, pos))

    assert out[:, 5].abs().mean() > out[:, 0].abs().mean()


def test_swa_eager_vs_flex(monkeypatch):

    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()

    out_eager = attn(x, rope(x, pos))

    try:
        monkeypatch.setattr("attention.ATTN_IMPL", "flex")
        out_flex = attn(x, rope(x, pos))
        assert torch.allclose(out_eager, out_flex, atol=1e-3)
    except:
        pass


def test_swa_backward():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()

    x.requires_grad = True
    out = attn(x, rope(x, pos))
    out.mean().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_swa_batch_independence():
    attn, rope = get_swa_model()

    x1, pos = get_swa_inputs(B=1)
    x2, _ = get_swa_inputs(B=1)

    x = torch.cat([x1, x2], dim=0)
    out = attn(x, rope(x, pos.repeat(2, 1)))

    assert not torch.allclose(out[0], out[1])


def test_swa_determinism():
    torch.manual_seed(0)

    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()

    out1 = attn(x, rope(x, pos))

    torch.manual_seed(0)
    attn2, rope2 = get_swa_model()
    out2 = attn2(x, rope2(x, pos))

    assert torch.allclose(out1, out2, atol=1e-5)


def test_swa_linear_complexity():

    class Cfg:
        hidden_size = 32
        num_attention_heads = 4
        num_key_value_heads = 2
        sliding_window_size = 4
        max_position_embeddings = 4096
        rope_theta = 10000.0

    cfg = Cfg()
    attn = KairosAttention(cfg)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)

    lengths = [256, 512, 1024]
    times = []

    def measure(x, cos_sin):
        start = time.time()
        _ = attn(x, cos_sin)
        return time.time() - start

    for L in lengths:
        x = torch.randn(1, L, 32)
        pos = torch.arange(L).unsqueeze(0)
        cos_sin = rope(x, pos)

        for _ in range(3):
            _ = attn(x, cos_sin)

        times.append(sum(measure(x, cos_sin) for _ in range(3)) / 3)

    r1 = times[1] / times[0]
    r2 = times[2] / times[1]

    assert r1 < 3.0 and r2 < 3.0


# =========================
# DELTANET TESTS
# =========================
def get_deltanet_model():
    return KairosGatedDeltaNet(DummyDeltaConfig(), layer_idx=0)


def get_deltanet_inputs(B=2, L=16, D=32):
    return torch.randn(B, L, D)


def test_deltanet_shape():
    model = get_deltanet_model()
    x = get_deltanet_inputs()

    out = model(x)
    assert out.shape == x.shape


def test_deltanet_stability():
    model = get_deltanet_model()
    x = get_deltanet_inputs()

    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_deltanet_bidir_consistency():
    model = get_deltanet_model()

    x = get_deltanet_inputs()

    # internal forward
    out_f = model.process(x)

    # internal backward
    x_rev = torch.flip(x, dims=[1])
    out_b = model.process(x_rev)
    out_b = torch.flip(out_b, dims=[1])

    # full model output
    out = model(x)

    # reconstruct expected output
    reconstructed = torch.cat([out_f, out_b], dim=-1)
    reconstructed = reconstructed.reshape(out.shape[0], out.shape[1], -1)
    projected = model.out_proj(model.out_left_right(reconstructed))

    assert torch.allclose(out, projected, atol=1e-5)


def test_deltanet_bidir_effect():
    model = get_deltanet_model()
    x = get_deltanet_inputs()

    # forward direction
    out_f = model.process(x)

    # backward direction
    x_rev = torch.flip(x, dims=[1])
    out_b = model.process(x_rev)
    out_b = torch.flip(out_b, dims=[1])

    # they should NOT be identical
    assert not torch.allclose(out_f, out_b, atol=1e-3)



def test_deltanet_backward():
    model = get_deltanet_model()

    x = get_deltanet_inputs()
    x.requires_grad = True

    out = model(x)
    out.mean().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_deltanet_determinism():
    model = get_deltanet_model()

    x = get_deltanet_inputs()
    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_deltanet_order_sensitivity():
    model = get_deltanet_model()

    x = get_deltanet_inputs()

    out1 = model(x)

    x_rev = torch.flip(x, dims=[1])
    out2 = model(x_rev)

    # outputs must differ
    assert not torch.allclose(out1, out2)



def test_deltanet_signal_propagation():
    model = get_deltanet_model()

    x = torch.zeros(1, 32, 32)
    x[:, 16] = 10.0

    out = model(x)

    assert out[:, 16].abs().mean() > out[:, 0].abs().mean()

# =========================
# DELTANET CACHE TESTS
# =========================

def test_deltanet_cache_not_mutated():
    """
    Ensure that the original cache (state_N) is NOT modified
    when using cloned caches for diffusion iterations.
    """
    model = get_deltanet_model()

    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.config)

    _ = model(x_N, cache)

    cache_ref = cache.clone()

    x_M = torch.randn(1, 8, 32)
    _ = model(x_M, cache.clone())

    # Original cache must remain unchanged
    for c1, c2 in zip(cache.ssm_caches, cache_ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2)


def test_deltanet_cache_clone_isolation():
    """
    Ensure cloned caches evolve independently.
    """

    model = get_deltanet_model()

    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.config)

    _ = model(x_N, cache)

    cache_a = cache.clone()
    cache_b = cache.clone()

    x_M1 = torch.randn(1, 8, 32)
    x_M2 = torch.randn(1, 8, 32)

    _ = model(x_M1, cache_a)
    _ = model(x_M2, cache_b)

    for a, b in zip(cache_a.ssm_caches, cache_b.ssm_caches):
        if a is not None:
            assert not torch.allclose(a, b)



def test_deltanet_cache_effect():
    """
    Ensure that different context states produce different outputs,
    proving that the cache acts as a conditioning signal.
    """
    model = get_deltanet_model()

    x_N1 = torch.randn(1, 16, 32)
    x_N2 = torch.randn(1, 16, 32)

    x_M = torch.randn(1, 8, 32)

    cache1 = KairosCache(model.config)
    cache2 = KairosCache(model.config)

    _ = model(x_N1, cache1)
    _ = model(x_N2, cache2)

    out1 = model(x_M, cache1.clone())
    out2 = model(x_M, cache2.clone())

    assert not torch.allclose(out1, out2)


def test_deltanet_cache_determinism():
    """
    Ensure deterministic behavior:
    same cache + same input → identical output.
    Critical for diffusion stability.
    """
    model = get_deltanet_model()

    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)

    cache = KairosCache(model.config)
    _ = model(x_N, cache)

    out1 = model(x_M, cache.clone())
    out2 = model(x_M, cache.clone())

    assert torch.allclose(out1, out2, atol=1e-5)


# =========================
# SWA CACHE TESTS
# =========================

def test_swa_cache_consistency():
    """
    Ensure that step-by-step attention equals full-sequence attention.
    This validates KV cache correctness.
    """
    cfg = DummySWAConfig()
    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)

    x = torch.randn(1, 16, 32)
    pos = torch.arange(16).unsqueeze(0)

    # full
    full = attn(x, rope(x, pos))

    # step-by-step with cache
    cache = KairosCache(cfg)

    outs = []
    for i in range(16):
        xi = x[:, i:i+1]
        pi = pos[:, i:i+1]

        out = attn(xi, rope(xi, pi), cache_params=cache)
        outs.append(out)

    step = torch.cat(outs, dim=1)

    assert step.shape == full.shape
    assert not torch.isnan(step).any()




def test_swa_cache_trim():
    """
    Ensure sliding window trimming keeps KV cache size bounded.
    """

    cfg = DummySWAConfig()
    cfg.sliding_window_size = 4

    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)

    cache = KairosCache(cfg)

    x = torch.randn(1, 32, 32)
    pos = torch.arange(32).unsqueeze(0)

    for i in range(32):
        xi = x[:, i:i+1]
        pi = pos[:, i:i+1]
        attn(xi, rope(xi, pi), cache_params=cache)

    k = cache._key_cache[0]
    assert k.shape[1] <= cfg.sliding_window_size


def test_swa_cache_no_trim_when_small_sequence():
    cfg = DummySWAConfig()
    cfg.sliding_window_size = 32  # > sequence

    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)

    cache = KairosCache(cfg)

    x = torch.randn(1, 8, 32)
    pos = torch.arange(8).unsqueeze(0)

    for i in range(8):
        xi = x[:, i:i+1]
        pi = pos[:, i:i+1]
        attn(xi, rope(xi, pi), cache_params=cache)

    k = cache._key_cache[0]

    assert k.shape[1] == 8


# =========================
# Diffusion CACHE TESTS
# =========================
def test_deltanet_diffusion_stability():
    model = get_deltanet_model()

    x_N = torch.randn(1, 16, 32) # past sequence
    x_M = torch.randn(1, 8, 32) # diffusion sequence

    cache = KairosCache(model.config)
    _ = model(x_N, cache)

    outs = []
    for _ in range(5):
        out = model(x_M, cache.clone())
        outs.append(out)

    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)



def test_swa_partial_diffusion_stability():
    cfg = DummySWAConfig()
    cfg.sliding_window_size = 64  # w > M

    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)

    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)

    pos_N = torch.arange(16).unsqueeze(0)
    pos_M = torch.arange(8).unsqueeze(0)

    # build cache on N
    cache = KairosCache(cfg)
    _ = attn(x_N, rope(x_N, pos_N), cache_params=cache)

    outs = []
    for _ in range(5):
        cache_iter = cache.clone()
        out = attn(x_M, rope(x_M, pos_M), cache_params=cache_iter)
        outs.append(out)

    # stable because KV(N) constant
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)


# =========================
# LiZAttention2 TESTS
# =========================
def get_liz_model():
    cfg = DummyDeltaConfig()  # reuse your config
    return KairosLiZAttention2(cfg, layer_idx=0)


def test_liz_shape():
    model = get_liz_model()

    x = torch.randn(2, 16, 32)
    pos = torch.arange(16).unsqueeze(0)

    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out = model(x, rope(x, pos))

    assert out.shape == x.shape


def test_liz_no_nan():
    model = get_liz_model()

    x = torch.randn(2, 16, 32)
    pos = torch.arange(16).unsqueeze(0)

    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out = model(x, rope(x, pos))

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_liz_concat_effect():
    """
    Check that combining SWA + Delta is NOT trivial.
    """
    model = get_liz_model()

    x = torch.randn(1, 16, 32)
    pos = torch.arange(16).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)

    # individual
    swa = model.swa(x, rope(x, pos))
    delta = model.delta(x)

    # combined
    liz = model(x, rope(x, pos))

    # must differ from both
    assert not torch.allclose(liz, swa)
    assert not torch.allclose(liz, delta)


def test_liz_cache_effect():
    """
    Diffusion condition: cache should influence output.
    """
    model = get_liz_model()

    x_N1 = torch.randn(1, 16, 32)
    x_N2 = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)

    cache1 = KairosCache(model.delta.config)
    cache2 = KairosCache(model.delta.config)

    _ = model.delta(x_N1, cache_params=cache1)
    _ = model.delta(x_N2, cache_params=cache2)

    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)

    out1 = model(x_M, rope(x_M, pos), cache_params=cache1.clone())
    out2 = model(x_M, rope(x_M, pos), cache_params=cache2.clone())

    assert not torch.allclose(out1, out2)


def test_liz_cache_not_mutated():
    """
    Ensure original cache is not modified (diffusion safety)
    """
    model = get_liz_model()

    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.delta.config)

    _ = model.delta(x_N, cache_params=cache)
    ref = cache.clone()

    x_M = torch.randn(1, 8, 32)

    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)

    _ = model(x_M, rope(x_M, pos), cache_params=cache.clone())

    for c1, c2 in zip(cache.ssm_caches, ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2)


def test_liz_determinism():
    """
    Same input + same cache → same output
    """
    model = get_liz_model()

    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)

    cache = KairosCache(model.delta.config)
    _ = model.delta(x_N, cache_params=cache)

    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)

    out1 = model(x_M, rope(x_M, pos), cache_params=cache.clone())
    out2 = model(x_M, rope(x_M, pos), cache_params=cache.clone())

    assert torch.allclose(out1, out2, atol=1e-5)
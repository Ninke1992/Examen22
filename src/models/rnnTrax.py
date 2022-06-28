import sys

import gin
from trax import layers as tl
from trax.layers import combinators as cb
from trax.layers.assert_shape import assert_shape

sys.path.insert(0, "..")


def last():
    # returns last element of hidden layer
    return tl.Fn("Last", lambda x: x[:, -1, :], n_out=1)


def avg_last():
    # returns average of hidden layer
    return tl.Fn("AvgLast", lambda x: x.mean(axis=-1), n_out=1)


@assert_shape("bld->bd")
def BaseNLPTrax(config: dict):
    model = cb.Serial(
        tl.Embedding(vocab_size=config["vocab"], d_feature=config["hidden_size"]),
        tl.GRU(n_units=config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        last(),
        tl.Dense(config["output_size"]),
    )
    return model


@assert_shape("bld->bd")
def NLPTrax2Layer(config: dict, mode="train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=config["vocab"], d_feature=config["hidden_size"]),
        tl.GRU(n_units=config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        tl.GRU(n_units=config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        last(),
        tl.Dense(config["output_size"]),
    )
    return model


@gin.configurable
@assert_shape("bld->bd")
def NLPTraxAvgLastConfig(
    units: int, vocab_size: int, dropout: float, output_size: int, mode="train"
):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=vocab_size, d_feature=units),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        avg_last(),
        tl.Dense(output_size),
    )
    return model


@gin.configurable
@assert_shape("bld->bd")
def NLPTraxCausalAttention(
    units: int,
    dropout: float,
    output_size: int,
    heads: int,
    vocab_size: int,
    mode="train",
):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=vocab_size, d_feature=units),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature=units, n_heads=heads, dropout=dropout),
        avg_last(),
        tl.Dense(output_size),
    )
    return model


@gin.configurable
@assert_shape("bld->bd")
def NLPTraxCausalAttentionOneGru(
    units: int,
    dropout: float,
    output_size: int,
    heads: int,
    vocab_size: int,
    mode="train",
):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=vocab_size, d_feature=units),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature=units, n_heads=heads, dropout=dropout),
        avg_last(),
        tl.Dense(output_size),
    )
    return model


@gin.configurable
@assert_shape("bd->bd")
def NLPTraxCausalAttentionOneGruLast(
    units: int,
    dropout: float,
    output_size: int,
    heads: int,
    vocab_size: int,
    mode="train",
):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=vocab_size, d_feature=units),
        tl.GRU(n_units=units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature=units, n_heads=heads, dropout=dropout),
        last(),
        tl.Dense(output_size),
    )
    return model

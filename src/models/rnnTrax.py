import sys

import gin
import jax.numpy as jnp
import numpy as np
from trax import layers as tl
from trax.shapes import signature
from trax.layers import combinators as cb
from trax.layers.assert_shape import assert_shape


sys.path.insert(0, "..")
from src.models.summary import summary

def Last():
    return tl.Fn("Last", lambda x: x[:, -1, :], n_out=1)

def AvgLast():
    return tl.Fn("AvgLast", lambda x: x.mean(axis=-1), n_out=1)

def BaseNLPTrax(config: dict):
    model = cb.Serial(
        tl.Embedding(vocab_size = config["vocab"], d_feature= config["hidden_size"]),
        tl.GRU(n_units=config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        Last(),
        tl.Dense(config["output_size"]),
    )
    return model

def NLPTrax2Layer(config: dict, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = config["vocab"], d_feature= config["hidden_size"]),
        tl.GRU(n_units = config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        tl.GRU(n_units = config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        Last(),
        tl.Dense(config["output_size"]),
    )
    return model

@gin.configurable
def NLPTraxAvgLastConfig(units: int, vocab_size: int, dropout: float, output_size: int, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = vocab_size, d_feature= units),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        AvgLast(),
        tl.Dense(output_size),
    )
    return model

def NLPTraxAvgLast(config: dict, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = config["vocab"], d_feature= config["hidden_size"]),
        tl.GRU(n_units = config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        tl.GRU(n_units = config["hidden_size"]),
        tl.Dropout(config["dropout"]),
        AvgLast(),
        tl.Dense(config["output_size"]),
    )
    return model

@gin.configurable
def NLPTraxCausalAttention(units: int, dropout: float, output_size: int, heads: int, vocab_size:int, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = vocab_size, d_feature= units),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature = units, n_heads = heads, dropout = dropout),
        AvgLast(),
        tl.Dense(output_size),
    )
    return model


@gin.configurable
def NLPTraxCausalAttentionOneGru(units: int, dropout: float, output_size: int, heads: int, vocab_size:int, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = vocab_size, d_feature= units),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature = units, n_heads = heads, dropout = dropout),
        AvgLast(),
        tl.Dense(output_size),
    )
    return model

@gin.configurable
def NLPTraxCausalAttentionOneGruLast(units: int, dropout: float, output_size: int, heads: int, vocab_size:int, mode = "train"):
    model = cb.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size = vocab_size, d_feature= units),
        tl.GRU(n_units = units),
        tl.Dropout(dropout),
        tl.CausalAttention(d_feature = units, n_heads = heads, dropout = dropout),
        Last(),
        tl.Dense(output_size),
    )
    return model

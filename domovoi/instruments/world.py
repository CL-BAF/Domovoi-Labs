"""Versioned synthetic worlds. The key holder belongs on the data side only."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, Iterator

BOS, EOS, HELDOUT, UNTRAINED, STRUCTURED, HOP, LINK = range(256, 263)
ATTRIBUTE = 264
DOMAIN = b"domovoi/world/v1\0"


@dataclass(frozen=True)
class Query:
    """The entire model-facing query. It contains neither labels nor a key."""

    identity: int
    attribute: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class Example:
    """A training record or an evaluator-side query/answer pair."""

    query: Query
    label: tuple[int, int]

    @property
    def training_tokens(self) -> tuple[int, ...]:
        return (*self.query.tokens, *self.label, EOS)


def _u16(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 65536:
        raise ValueError("Expected an unsigned 16-bit integer.")
    return value.to_bytes(2, "big")


class World:
    """Keep this object and answer records out of the prediction process."""

    def __init__(self, root_key: bytes, seed: int, identities: int, attributes: int = 4):
        if not isinstance(root_key, bytes) or len(root_key) != 32:
            raise ValueError("Provide a 32-byte root key; generate it outside the model.")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Expected an unsigned 64-bit seed.")
        if type(identities) is not int or not 2 <= identities <= 65536:
            raise ValueError("A world needs 2 through 65,536 identities.")
        if type(attributes) is not int or not 1 <= attributes <= 32:
            raise ValueError("A world needs 1 through 32 attributes.")
        context = (seed.to_bytes(8, "big") + identities.to_bytes(4, "big") +
                   _u16(attributes))
        self._key = hmac.digest(root_key, DOMAIN + b"key\0" + context, "sha256")
        self.seed = seed
        self.identities = identities
        self.attributes = attributes
        self.id = hashlib.sha256(DOMAIN + b"commitment\0" + self._key).hexdigest()

    def __reduce_ex__(self, protocol):
        raise TypeError("A keyed world must not be serialized as a model input.")

    @property
    def fact_count(self) -> int:
        return self.identities * self.attributes

    def _draw(self, domain: bytes, message: bytes) -> bytes:
        return hmac.digest(self._key, DOMAIN + domain + b"\0" + message, "sha256")

    def example(self, identity: int, attribute: int, kind: str = "training") -> Example:
        if type(identity) is not int or not 0 <= identity < self.identities:
            raise ValueError("Identity is outside this world.")
        if type(attribute) is not int or not 0 <= attribute < self.attributes:
            raise ValueError("Attribute is outside this world.")
        digits = tuple(_u16(identity))
        tag = ATTRIBUTE + attribute
        message = _u16(identity) + _u16(attribute)
        label = self._draw(b"random-fact", message)[:2]
        if kind == "training":
            tokens = (BOS, *digits, tag)
        elif kind == "heldout":
            tokens = (BOS, HELDOUT, tag, *digits)
        elif kind == "untrained":
            tokens = (BOS, UNTRAINED, tag, *digits)
            label = self._draw(b"untrained", message)[:2]
        elif kind in ("structured-train", "structured-heldout"):
            # Split by identity before generating records. Rule labels do not enter K.
            heldout = identity % 5 == 0
            if heldout != (kind == "structured-heldout"):
                raise ValueError("Structured identity is in the other split.")
            tokens = (BOS, STRUCTURED, tag, *digits)
            a, b, c = (int.from_bytes(self._draw(b"rule", _u16(attribute))[i:i + 2], "big")
                       for i in (0, 2, 4))
            label = _u16(((a | 1) * digits[0] + b * digits[1] + c) % 65536)
        elif kind in ("link", "two-hop"):
            target = self.link_target(identity)
            if kind == "link":
                if attribute != 0:
                    raise ValueError("Link records use attribute zero.")
                tokens, label = (BOS, LINK, *digits), _u16(target)
            else:
                tokens = (BOS, HOP, tag, *digits)
                label = self._draw(b"random-fact", _u16(target) + _u16(attribute))[:2]
        else:
            raise ValueError(f"Unknown synthetic query kind {kind!r}.")
        return Example(Query(identity, attribute, tokens), tuple(label))

    def link_target(self, identity: int) -> int:
        if type(identity) is not int or not 0 <= identity < self.identities:
            raise ValueError("Identity is outside this world.")
        # Uniform selection excluding self, using rejection rather than modulo bias.
        population = self.identities - 1
        limit = 2**32 - (2**32 % population)
        for counter in range(2**32):
            value = int.from_bytes(self._draw(
                b"link", _u16(identity) + counter.to_bytes(4, "big"))[:4], "big")
            if value < limit:
                other = value % population
                return other + (other >= identity)
        raise RuntimeError("PRF link rejection counter exhausted.")

    def instrument_partitions(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Every sixteenth trained identity calibrates; the remainder is scored."""
        return (tuple(range(0, self.identities, 16)),
                tuple(i for i in range(self.identities) if i % 16))

    def random_facts(self, kind: str = "training",
                     identities: Iterable[int] | None = None) -> Iterator[Example]:
        if kind not in ("training", "heldout", "untrained"):
            raise ValueError("The random fact population cannot include derived probes.")
        for identity in range(self.identities) if identities is None else identities:
            for attribute in range(self.attributes):
                yield self.example(identity, attribute, kind)

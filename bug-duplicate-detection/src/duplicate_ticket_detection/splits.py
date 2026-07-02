from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from duplicate_ticket_detection.dataset import TicketRecord


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    train: list[TicketRecord]
    test: list[TicketRecord]


def train_test_split_tickets(
    tickets: Iterable[TicketRecord],
    *,
    test_size: float = 0.2,
    seed: int = 13,
) -> DatasetSplit:
    """Split tickets so duplicate groups keep at least one member in train and one in test."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = random.Random(seed)
    groups, singletons = _partition_by_duplicate_group(tickets)
    train: list[TicketRecord] = []
    test: list[TicketRecord] = []

    for members in groups.values():
        shuffled = list(members)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        n_test = min(n_test, len(shuffled) - 1)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    shuffled_singletons = list(singletons)
    rng.shuffle(shuffled_singletons)
    singleton_test_count = round(len(shuffled_singletons) * test_size)
    test.extend(shuffled_singletons[:singleton_test_count])
    train.extend(shuffled_singletons[singleton_test_count:])

    return DatasetSplit(name=f"train_test_seed_{seed}", train=train, test=test)


def kfold_ticket_splits(
    tickets: Iterable[TicketRecord],
    *,
    n_splits: int = 5,
    seed: int = 13,
) -> list[DatasetSplit]:
    """Create folds for duplicate retrieval.

    Duplicate groups are split by members, not as whole groups. This keeps a duplicate
    counterpart in the training/candidate set whenever a held-out query is evaluated.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    rng = random.Random(seed)
    groups, singletons = _partition_by_duplicate_group(tickets)
    fold_tests: list[list[TicketRecord]] = [[] for _ in range(n_splits)]

    for group_index, members in enumerate(groups.values()):
        shuffled = list(members)
        rng.shuffle(shuffled)
        fold_offset = group_index % n_splits
        for index, ticket in enumerate(shuffled):
            fold_tests[(fold_offset + index) % n_splits].append(ticket)

    shuffled_singletons = list(singletons)
    rng.shuffle(shuffled_singletons)
    for index, ticket in enumerate(shuffled_singletons):
        fold_tests[index % n_splits].append(ticket)

    all_tickets = list(tickets)
    splits: list[DatasetSplit] = []
    for fold_index, test in enumerate(fold_tests, start=1):
        test_ids = {ticket.ticket_id for ticket in test}
        train = [ticket for ticket in all_tickets if ticket.ticket_id not in test_ids]
        splits.append(DatasetSplit(name=f"fold_{fold_index}", train=train, test=test))

    return splits


def _partition_by_duplicate_group(
    tickets: Iterable[TicketRecord],
) -> tuple[dict[str, list[TicketRecord]], list[TicketRecord]]:
    groups: dict[str, list[TicketRecord]] = {}
    singletons: list[TicketRecord] = []

    for ticket in tickets:
        if ticket.duplicate_group:
            groups.setdefault(ticket.duplicate_group, []).append(ticket)
        else:
            singletons.append(ticket)

    duplicate_groups = {group: members for group, members in groups.items() if len(members) >= 2}
    for group, members in groups.items():
        if group not in duplicate_groups:
            singletons.extend(members)

    return duplicate_groups, singletons

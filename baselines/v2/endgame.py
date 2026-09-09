"""King, pawn and king: the whole three-man space, solved once at import.

Pawn endings turn on geometry rather than on depth. Whether K+P beats a lone
king is a question of opposition and key squares, and a search that cannot see
all the way to promotion scores every one of those positions alike, as "a pawn
up". That is how a won ending gets traded into a drawn one, and how a drawn one
gets defended as though it were already lost.

So we solve it exhaustively instead. A state is (side to move, white king,
black king, white pawn), and every state is either a win for the side with the
pawn or a draw -- a bare king has nothing to win with. Values come from
iterating the game-theoretic definition to a fixed point: with the pawn's owner
to move a position is won if any move reaches a won position, and drawn only if
every move fails; with the bare king to move it is drawn if any move reaches a
draw, and won only if every move loses. Repeat until a pass changes nothing.

Each pass is vectorised over all 262,144 positions per side to move at once
with numpy, which is what keeps the build inside the 90 second import budget
instead of on the clock. Positions with a black pawn are mirrored onto the
table by `probe`.
"""

from __future__ import annotations

import time
from typing import Any

import chess
import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

UNKNOWN = 0
DRAW = 1
WIN = 2

# States per side to move, indexed white_king * 4096 + black_king * 64 + pawn.
SIDE_STATES = 64 * 64 * 64
KING_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
# The longest forced win here is around thirty moves, so this many passes is
# slack, not a real bound. Reaching it would leave undecided states, which
# `probe` reads as draws.
MAX_PASSES = 256

_INDEX = np.arange(SIDE_STATES, dtype=np.int32)
_WHITE_KING = _INDEX >> 12
_BLACK_KING = (_INDEX >> 6) & 63
_PAWN = _INDEX & 63


def _files(squares: Array) -> Array:
    return squares & 7


def _ranks(squares: Array) -> Array:
    return squares >> 3


def _king_distance(a: Array, b: Array) -> Array:
    """Chebyshev distance: how many king moves apart two squares are."""
    return np.maximum(np.abs(_files(a) - _files(b)), np.abs(_ranks(a) - _ranks(b)))


def _step(squares: Array, file_delta: int, rank_delta: int) -> Array:
    """One king step from each square, or -1 where the step leaves the board."""
    file_ = _files(squares) + file_delta
    rank = _ranks(squares) + rank_delta
    on_board = (file_ >= 0) & (file_ < 8) & (rank >= 0) & (rank < 8)
    return np.where(on_board, rank * 8 + file_, -1).astype(np.int32)


def _attacked_by_pawn(pawn: Array, square: Array) -> Array:
    diagonal = np.abs(_files(square) - _files(pawn)) == 1
    return np.asarray((_ranks(square) == _ranks(pawn) + 1) & diagonal)


def _index(white_king: Array, black_king: Array, pawn: Array, live: Array) -> Array:
    """State numbers for the squares a move lands on, reading 0 where it is not a move.

    Squares off the board are -1 and pawn pushes can run past the last rank, so
    every component is folded to a real square before it is combined. Those slots
    are masked out by the caller anyway; this only keeps the lookup in bounds.
    """
    index = np.maximum(white_king, 0) * 4096 + np.maximum(black_king, 0) * 64
    return np.where(live, index + np.minimum(np.maximum(pawn, 0), 63), 0)


def _build() -> tuple[bytes, int, int]:
    pawn_rank = _ranks(_PAWN)
    # A pawn sits on ranks 2..7: rank 1 is impossible, and rank 8 has already
    # promoted out of this table.
    on_board = (
        (pawn_rank >= 1)
        & (pawn_rank <= 6)
        & (_WHITE_KING != _BLACK_KING)
        & (_WHITE_KING != _PAWN)
        & (_BLACK_KING != _PAWN)
        & (_king_distance(_WHITE_KING, _BLACK_KING) > 1)
    )
    gives_check = _attacked_by_pawn(_PAWN, _BLACK_KING)
    # With the pawn's owner to move the other side has just moved, so it cannot
    # have left itself in check.
    strong_legal = on_board & ~gives_check
    weak_legal = on_board

    # Pushing to the last rank leaves this table, so it is scored on the spot:
    # a promotion wins unless the new queen can simply be taken, which leaves
    # bare kings.
    promotes = pawn_rank == 6
    push = _PAWN + 8
    push_clear = (push != _WHITE_KING) & (push != _BLACK_KING)
    queen_safe = (_king_distance(_BLACK_KING, push) > 1) | (_king_distance(_WHITE_KING, push) == 1)
    # Every push mask carries `on_board`, which keeps the pawn on ranks 2..7 and
    # so keeps the square it pushes to on the board as well.
    promotion = promotes & push_clear & on_board
    promotion_wins = promotion & queen_safe
    single_push = ~promotes & push_clear & on_board
    double_target = _PAWN + 16
    double_push = (
        (pawn_rank == 1)
        & push_clear
        & on_board
        & (double_target != _WHITE_KING)
        & (double_target != _BLACK_KING)
    )
    pawn_defended = _king_distance(_WHITE_KING, _PAWN) == 1

    # The move graph never changes, so build it once: for every move, a mask of
    # the states where it is legal and the state it lands in. The passes below
    # are then nothing but table lookups, which is most of the build time back.
    strong_moves: list[tuple[Array, Array]] = []
    strong_has_move = promotion.copy()
    for file_delta, rank_delta in KING_STEPS:
        target = _step(_WHITE_KING, file_delta, rank_delta)
        moves = (target >= 0) & (target != _PAWN) & (_king_distance(target, _BLACK_KING) > 1)
        strong_has_move |= moves
        strong_moves.append((moves, _index(target, _BLACK_KING, _PAWN, moves)))
    for target, allowed in ((push, single_push), (double_target, double_push)):
        strong_has_move |= allowed
        strong_moves.append((allowed, _index(_WHITE_KING, _BLACK_KING, target, allowed)))

    weak_moves: list[tuple[Array, Array]] = []
    weak_has_move = np.zeros(SIDE_STATES, dtype=bool)
    # Taking the pawn leaves bare kings, a draw that is not in the table.
    takes_pawn = np.zeros(SIDE_STATES, dtype=bool)
    for file_delta, rank_delta in KING_STEPS:
        target = _step(_BLACK_KING, file_delta, rank_delta)
        onto_pawn = target == _PAWN
        moves = (
            (target >= 0)
            & (_king_distance(target, _WHITE_KING) > 1)
            & (onto_pawn | ~_attacked_by_pawn(_PAWN, np.maximum(target, 0)))
            & (~onto_pawn | ~pawn_defended)
        )
        weak_has_move |= moves
        takes_pawn |= moves & onto_pawn
        quiet = moves & ~onto_pawn
        weak_moves.append((quiet, _index(_WHITE_KING, target, _PAWN, quiet)))

    # With no move at all it is stalemate unless the pawn is giving check, and
    # then it is mate. Stalemate is a draw; the WIN default covers mate.
    stalemate = ~weak_has_move & ~gives_check

    strong = np.where(strong_legal, UNKNOWN, DRAW).astype(np.uint8)
    weak = np.where(weak_legal, UNKNOWN, DRAW).astype(np.uint8)

    passes = 0
    for passes in range(1, MAX_PASSES + 1):  # noqa: B007
        # The side with the pawn needs one won child; the bare king needs one
        # drawn child. A state only ever goes from undecided to decided, so
        # reading the fresher table in the second half just gets us there sooner.
        has_win = promotion_wins.copy()
        undecided = np.zeros(SIDE_STATES, dtype=bool)
        for moves, index in strong_moves:
            child = weak[index]
            has_win |= moves & (child == WIN)
            undecided |= moves & (child == UNKNOWN)
        # No move at all is stalemate, which the DRAW default already covers.
        next_strong = np.where(has_win, WIN, np.where(undecided, UNKNOWN, DRAW))
        next_strong = np.where(strong_legal, next_strong, DRAW).astype(np.uint8)

        has_draw = takes_pawn.copy()
        undecided = np.zeros(SIDE_STATES, dtype=bool)
        for moves, index in weak_moves:
            child = next_strong[index]
            has_draw |= moves & (child == DRAW)
            undecided |= moves & (child == UNKNOWN)
        next_weak = np.where(has_draw | stalemate, DRAW, np.where(undecided, UNKNOWN, WIN))
        next_weak = np.where(weak_legal, next_weak, DRAW).astype(np.uint8)

        settled = np.array_equal(next_strong, strong) and np.array_equal(next_weak, weak)
        strong, weak = next_strong, next_weak
        if settled:
            break

    table = np.concatenate((strong, weak)) == WIN
    return table.tobytes(), passes, int(np.count_nonzero(table))


_started_at = time.monotonic()
_TABLE, _PASSES, _WON = _build()
print(
    f"kpk bitbase {_WON:,} won of {2 * SIDE_STATES:,} states, "
    f"{_PASSES} passes, {time.monotonic() - _started_at:.1f}s"
)


def is_pawn_ending(board: chess.Board) -> bool:
    """Whether this is king and one pawn against a bare king."""
    return chess.popcount(board.occupied) == 3 and board.pawns != 0


def probe(board: chess.Board) -> chess.Color | None:
    """The colour that wins a K+P vs K position, or None when it is a draw.

    The caller has to have established that the position really is that ending;
    `is_pawn_ending` is the check.
    """
    strong = chess.WHITE if board.pawns & board.occupied_co[chess.WHITE] else chess.BLACK
    pawn = chess.msb(board.pawns)
    strong_king = chess.msb(board.kings & board.occupied_co[strong])
    weak_king = chess.msb(board.kings & board.occupied_co[not strong])
    if strong == chess.BLACK:
        # The table only holds white pawns, so mirror the board top to bottom.
        pawn = chess.square_mirror(pawn)
        strong_king = chess.square_mirror(strong_king)
        weak_king = chess.square_mirror(weak_king)
    offset = 0 if board.turn == strong else SIDE_STATES
    return strong if _TABLE[offset + strong_king * 4096 + weak_king * 64 + pawn] else None

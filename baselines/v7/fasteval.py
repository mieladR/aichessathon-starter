"""The evaluation, compiled.

`agent.evaluate` is the readable definition of what a position is worth and it
stays the authority: this module is a line-for-line translation of its main body
into something numba can compile, and it is required to agree with it exactly.
`agent` checks that agreement at import over a spread of positions and falls
back to the Python version if it ever fails, so a numba that behaves differently
on the platform costs speed and never correctness.

Why bother: the evaluation is about forty per cent of search time once the move
generator stops being the bottleneck, and every node pays it. Compiling it buys
most of a ply, and a ply is worth more than any evaluation term we could add.

Everything here works on the six piece bitboards plus the two colour masks,
which is exactly what python-chess already keeps, so calling it costs no
marshalling beyond passing nine integers.

numba compiles on first call and that first call costs far more than the move it
would be part of, so every function is given an explicit signature. That makes
numba compile it at import, inside the 90 second budget, rather than on the
clock. Nothing here is cached to disk: the filesystem is read-only and /tmp is
wiped between games, so `cache=True` would buy nothing.
"""

from __future__ import annotations

import numpy as np
from numba import boolean, int64, njit, uint64

# ---------------------------------------------------------------------------
# Tables. Built here in Python from plain integers, then frozen into numpy
# arrays that the compiled code indexes directly.
# ---------------------------------------------------------------------------

U0 = np.uint64(0)
U1 = np.uint64(1)
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

BB_SQUARE = np.array([1 << square for square in range(64)], dtype=np.uint64)
RANK_BB = np.array([0xFF << (8 * rank) for rank in range(8)], dtype=np.uint64)
FILE_BB_RAW = [0x0101_0101_0101_0101 << file for file in range(8)]
FILE_BB = np.array(FILE_BB_RAW, dtype=np.uint64)
ADJACENT_FILES = np.array(
    [
        (FILE_BB_RAW[file - 1] if file > 0 else 0) | (FILE_BB_RAW[file + 1] if file < 7 else 0)
        for file in range(8)
    ],
    dtype=np.uint64,
)

# Popcount by 16-bit chunks: four lookups beats any loop, and 64 KB is nothing.
POP16 = np.array([bin(value).count("1") for value in range(1 << 16)], dtype=np.int64)

# De Bruijn multiplication maps an isolated low bit onto a unique six-bit index.
# The table is derived from the constant rather than transcribed, so it cannot
# disagree with it.
DEBRUIJN = 0x03F7_9D71_B4CB_0A89
DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _square in range(64):
    DEBRUIJN_INDEX[(((1 << _square) * DEBRUIJN) & MASK64) >> 58] = _square

KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
for _square in range(64):
    _file, _rank = _square & 7, _square >> 3
    _knight = 0
    for _df, _dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        _f, _r = _file + _df, _rank + _dr
        if 0 <= _f < 8 and 0 <= _r < 8:
            _knight |= 1 << (_r * 8 + _f)
    KNIGHT_ATTACKS[_square] = _knight
    _king = 0
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == 0 and _dr == 0:
                continue
            _f, _r = _file + _df, _rank + _dr
            if 0 <= _f < 8 and 0 <= _r < 8:
                _king |= 1 << (_r * 8 + _f)
    KING_ATTACKS[_square] = _king


def install(
    mg_white: list[list[int]],
    eg_white: list[list[int]],
    mg_black: list[list[int]],
    eg_black: list[list[int]],
    phase_weight: tuple[int, ...],
    forward_file: list[list[int]],
    passed_mask: list[list[int]],
    shield_mask: list[list[int]],
    king_danger: tuple[int, ...],
    mobility_mg: tuple[int, ...],
    mobility_eg: tuple[int, ...],
    king_attack_weight: tuple[int, ...],
) -> None:
    """Take the tables `agent` already built, so there is one source for each."""
    global MG_W, EG_W, MG_B, EG_B, PHASE_W, FORWARD, PASSED, SHIELD
    global DANGER, MOB_MG, MOB_EG, KATT, score
    MG_W = np.array(mg_white, dtype=np.int64)
    EG_W = np.array(eg_white, dtype=np.int64)
    MG_B = np.array(mg_black, dtype=np.int64)
    EG_B = np.array(eg_black, dtype=np.int64)
    PHASE_W = np.array(phase_weight, dtype=np.int64)
    FORWARD = np.array(forward_file, dtype=np.uint64)
    PASSED = np.array(passed_mask, dtype=np.uint64)
    SHIELD = np.array(shield_mask, dtype=np.uint64)
    DANGER = np.array(king_danger, dtype=np.int64)
    MOB_MG = np.array(mobility_mg, dtype=np.int64)
    MOB_EG = np.array(mobility_eg, dtype=np.int64)
    KATT = np.array(king_attack_weight, dtype=np.int64)
    # Compile only now. numba resolves a global it reads into a compile-time
    # constant, so compiling `_score` at decoration time would freeze the empty
    # placeholders below into it. Doing it here, with the real tables in place,
    # is also what puts the compile inside the import budget rather than on the
    # clock -- an explicit signature makes numba compile on the spot instead of
    # waiting for a first call, and pins the arguments to unsigned 64-bit, where
    # a bitboard complement means what it does in Python.
    score = njit(_SIGNATURE, cache=False, nogil=True)(_score)


# Scalar weights. Plain ints, so numba folds them in as constants. `agent`
# asserts these against its own at import; they are duplicated rather than
# passed because a compiled constant cannot be a module attribute lookup.
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 24, 46
ROOK_OPEN_MG, ROOK_OPEN_EG = 26, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 11, 6
DOUBLED_MG, DOUBLED_EG = -9, -22
ISOLATED_MG, ISOLATED_EG = -13, -16
PHALANX_MG, PHALANX_EG = 6, 4
PASSED_MG = (0, 4, 6, 14, 30, 58, 96, 0)
PASSED_EG = (0, 12, 20, 36, 64, 106, 170, 0)
KING_SHIELD = 9
TEMPO = 10
MAX_PHASE = 24
MAX_DANGER_INDEX = 95

# Placeholders; _install replaces them and only then compiles `_score`.
score = None
MG_W = EG_W = MG_B = EG_B = PHASE_W = np.zeros((7, 64), dtype=np.int64)
FORWARD = PASSED = SHIELD = np.zeros((2, 64), dtype=np.uint64)
DANGER = MOB_MG = MOB_EG = KATT = np.zeros(96, dtype=np.int64)


@njit(int64(uint64), cache=False, nogil=True)
def _popcount(bb: int) -> int:
    return (
        POP16[bb & np.uint64(0xFFFF)]
        + POP16[(bb >> np.uint64(16)) & np.uint64(0xFFFF)]
        + POP16[(bb >> np.uint64(32)) & np.uint64(0xFFFF)]
        + POP16[bb >> np.uint64(48)]
    )


@njit(int64(uint64), cache=False, nogil=True)
def _lsb_index(bb: int) -> int:
    """The square number of the lowest set bit."""
    isolated = bb & (U0 - bb)
    return DEBRUIJN_INDEX[(isolated * np.uint64(DEBRUIJN)) >> np.uint64(58)]


@njit(uint64(int64, uint64), cache=False, nogil=True)
def _diagonal_attacks(square: int, occupied: int) -> int:
    attacks = U0
    file0 = square & 7
    rank0 = square >> 3
    for step_file, step_rank in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        file = file0 + step_file
        rank = rank0 + step_rank
        while 0 <= file < 8 and 0 <= rank < 8:
            bit = U1 << np.uint64(rank * 8 + file)
            attacks |= bit
            if occupied & bit:
                break
            file += step_file
            rank += step_rank
    return attacks


@njit(uint64(int64, uint64), cache=False, nogil=True)
def _straight_attacks(square: int, occupied: int) -> int:
    attacks = U0
    file0 = square & 7
    rank0 = square >> 3
    for step_file, step_rank in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        file = file0 + step_file
        rank = rank0 + step_rank
        while 0 <= file < 8 and 0 <= rank < 8:
            bit = U1 << np.uint64(rank * 8 + file)
            attacks |= bit
            if occupied & bit:
                break
            file += step_file
            rank += step_rank
    return attacks


@njit(uint64(int64, int64, uint64), cache=False, nogil=True)
def _piece_attacks(piece_type: int, square: int, occupied: int) -> int:
    if piece_type == 2:
        return KNIGHT_ATTACKS[square]
    if piece_type == 3:
        return _diagonal_attacks(square, occupied)
    if piece_type == 4:
        return _straight_attacks(square, occupied)
    return _diagonal_attacks(square, occupied) | _straight_attacks(square, occupied)


_SIGNATURE = int64(
    uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, boolean
)


def _score(
    pawns: int,
    knights: int,
    bishops: int,
    rooks: int,
    queens: int,
    kings: int,
    white: int,
    black: int,
    turn: bool,
) -> int:
    """Exactly what `agent.evaluate` computes for an ordinary position."""
    occupied = white | black
    mg = 0
    eg = 0
    phase = 0

    # Material and piece-square tables, plus the phase count that blends them.
    for piece_type in range(1, 7):
        if piece_type == 1:
            pieces = pawns
        elif piece_type == 2:
            pieces = knights
        elif piece_type == 3:
            pieces = bishops
        elif piece_type == 4:
            pieces = rooks
        elif piece_type == 5:
            pieces = queens
        else:
            pieces = kings
        weight = PHASE_W[piece_type]
        remaining = pieces & white
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            mg += MG_W[piece_type, square]
            eg += EG_W[piece_type, square]
            phase += weight
        remaining = pieces & black
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            mg -= MG_B[piece_type, square]
            eg -= EG_B[piece_type, square]
            phase += weight

    if _popcount(bishops & white) >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    if _popcount(bishops & black) >= 2:
        mg -= BISHOP_PAIR_MG
        eg -= BISHOP_PAIR_EG

    # Pawn structure: doubled, isolated, connected and passed.
    white_pawns = pawns & white
    black_pawns = pawns & black
    for colour in range(2):
        if colour == 1:
            own = white_pawns
            enemy = black_pawns
            sign = 1
        else:
            own = black_pawns
            enemy = white_pawns
            sign = -1
        remaining = own
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            file = square & 7
            rank = square >> 3
            relative = rank if colour == 1 else 7 - rank
            neighbours = own & ADJACENT_FILES[file]
            if not neighbours:
                mg += sign * ISOLATED_MG
                eg += sign * ISOLATED_EG
            elif neighbours & RANK_BB[rank]:
                mg += sign * PHALANX_MG
                eg += sign * PHALANX_EG
            if own & FORWARD[colour, square]:
                mg += sign * DOUBLED_MG
                eg += sign * DOUBLED_EG
            elif not enemy & PASSED[colour, square]:
                mg += sign * PASSED_MG[relative]
                eg += sign * PASSED_EG[relative]

    # Rooks want files their own pawns are not standing on, and ideally files
    # nobody's pawns are standing on.
    for colour in range(2):
        if colour == 1:
            remaining = rooks & white
            own_pawns = white_pawns
            sign = 1
        else:
            remaining = rooks & black
            own_pawns = black_pawns
            sign = -1
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            file_mask = FILE_BB[square & 7]
            if not pawns & file_mask:
                mg += sign * ROOK_OPEN_MG
                eg += sign * ROOK_OPEN_EG
            elif not own_pawns & file_mask:
                mg += sign * ROOK_SEMI_MG
                eg += sign * ROOK_SEMI_EG

    # Mobility and king safety share one pass over the pieces.
    white_king = _lsb_index(kings & white)
    black_king = _lsb_index(kings & black)
    white_zone = KING_ATTACKS[black_king] | BB_SQUARE[black_king]
    black_zone = KING_ATTACKS[white_king] | BB_SQUARE[white_king]
    white_units = 0
    white_attackers = 0
    black_units = 0
    black_attackers = 0
    for piece_type in range(2, 6):
        if piece_type == 2:
            pieces = knights
        elif piece_type == 3:
            pieces = bishops
        elif piece_type == 4:
            pieces = rooks
        else:
            pieces = queens
        mobility_mg = MOB_MG[piece_type]
        mobility_eg = MOB_EG[piece_type]
        attack_weight = KATT[piece_type]
        remaining = pieces & white
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            attacks = _piece_attacks(piece_type, square, occupied)
            free = _popcount(attacks & ~white)
            mg += mobility_mg * free
            eg += mobility_eg * free
            near_king = _popcount(attacks & white_zone)
            if near_king:
                white_units += attack_weight * near_king
                white_attackers += 1
        remaining = pieces & black
        while remaining:
            square = _lsb_index(remaining)
            remaining &= remaining - U1
            attacks = _piece_attacks(piece_type, square, occupied)
            free = _popcount(attacks & ~black)
            mg -= mobility_mg * free
            eg -= mobility_eg * free
            near_king = _popcount(attacks & black_zone)
            if near_king:
                black_units += attack_weight * near_king
                black_attackers += 1

    if white_attackers >= 2:
        index = white_units if white_units < MAX_DANGER_INDEX else MAX_DANGER_INDEX
        mg += DANGER[index]
    if black_attackers >= 2:
        index = black_units if black_units < MAX_DANGER_INDEX else MAX_DANGER_INDEX
        mg -= DANGER[index]
    white_shield = _popcount(white_pawns & SHIELD[1, white_king])
    black_shield = _popcount(black_pawns & SHIELD[0, black_king])
    mg += KING_SHIELD * (white_shield if white_shield < 3 else 3)
    mg -= KING_SHIELD * (black_shield if black_shield < 3 else 3)

    if phase > MAX_PHASE:
        phase = MAX_PHASE
    # Truncating division, matching the Python original: flooring would make a
    # position and its colour-swapped mirror differ by a centipawn.
    tapered = int64((mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE)
    return (tapered if turn else -tapered) + TEMPO

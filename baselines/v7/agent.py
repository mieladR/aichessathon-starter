"""AI Chessathon submission.

A negamax searcher with alpha-beta, iterative deepening on aspiration windows,
principal-variation search, a transposition table, null-move pruning, reverse
futility pruning, late move reductions, check extensions, and a quiescence
search that only plays out captures a static exchange evaluation says are worth
making.

The evaluation is material and piece-square tables tapered between middlegame
and endgame weights, plus pawn structure, piece mobility, rooks on open files,
the bishop pair and a king-safety term, with enough exact endgame knowledge to
finish a won game: a lone king is driven to the edge, and king-and-pawn against
king is read off the solved table in `endgame`.

Import time runs once per game (see get_move's docstring for the per-call
contract); the module-level tables below are read-only after that point, and
the mutable state further down (transposition table, killers, history, game
history, clock observations) is what is allowed to persist between our own
moves within a single game.
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import Any

import chess

import endgame

# The compiled evaluation is an optimisation, not a dependency. If numba is
# missing or will not compile on the machine we are handed, the Python
# evaluation below is still correct, and a slower correct agent beats no agent.
try:
    import fasteval

    _evaluator: Any = fasteval
except Exception as _exc:
    _evaluator = None
    print(f"fasteval unavailable, using the Python evaluation: {_exc!r}")

# Alpha-beta recurses one Python frame per ply, and check extensions on top of
# quiescence can stack a hundred of them on a forcing line. The default limit
# would probably hold, but a RecursionError inside get_move loses the game
# outright, so buy headroom we should never need.
sys.setrecursionlimit(20_000)

# ---------------------------------------------------------------------------
# Evaluation
#
# A position's value starts as "material plus piece-square tables", computed
# twice -- once with middlegame weights, once with endgame weights -- and
# blended by how much material is left on the board. The idea: a knight belongs
# on a central outpost, a rook on an open file, a king behind its pawns while
# queens are on but marching up the board once they are gone. A single static
# table cannot represent both, so we keep two and fade between them.
#
# Values are in centipawns (100 = one pawn) and follow the widely-taught
# "PeSTO" tapered scheme. Everything after the tables is the knowledge a table
# indexed by one square cannot hold, because it depends on where the *other*
# pieces are: whether a pawn is passed, whether a rook has an open file, how
# many squares a bishop actually has, how exposed a king is.
# ---------------------------------------------------------------------------

MG_VALUE = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}
EG_VALUE = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}
# How much each piece counts towards "how far into the game are we", indexed by
# piece type. A full set of minor and major pieces sums to 24.
PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24

# Driving a lone king to the edge. Material and the piece-square tables score a
# bare KQ vs K the same wherever the defending king stands, so a search that is
# already winning has no gradient to walk down: it shuffles until the fifty-move
# rule takes the win away. These two terms supply one -- push the lone king
# outward, and bring our own king up to it.
EDGE_WEIGHT = 30
APPROACH_WEIGHT = 12
CENTRE_DISTANCE = [
    int(max(abs(chess.square_file(square) - 3.5), abs(chess.square_rank(square) - 3.5)) - 0.5)
    for square in chess.SQUARES
]
# King and pawn against king is solved outright in `endgame`, so a won one is
# scored well clear of any ordinary material edge and a drawn one scores zero.
PAWN_ENDING_WIN = 600
PAWN_ENDING_PUSH = 20
PAWN_ENDING_ESCORT = 8

# Published a8->h1, one row per rank starting from the top of a printed board.
_RAW_PST: dict[int, tuple[list[int], list[int]]] = {
    chess.PAWN: (
        [0, 0, 0, 0, 0, 0, 0, 0,
         98, 134, 61, 95, 68, 126, 34, -11,
         -6, 7, 26, 31, 65, 56, 25, -20,
         -14, 13, 6, 21, 23, 12, 17, -23,
         -27, -2, -5, 12, 17, 6, 10, -25,
         -26, -4, -4, -10, 3, 3, 33, -12,
         -35, -1, -20, -23, -15, 24, 38, -22,
         0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0,
         178, 173, 158, 134, 147, 132, 165, 187,
         94, 100, 85, 67, 56, 53, 82, 84,
         32, 24, 13, 5, -2, 4, 17, 17,
         13, 9, -3, -7, -7, -8, 3, -1,
         4, 7, -6, 1, 0, -5, -1, -8,
         13, 8, 8, 10, 13, 0, 2, -7,
         0, 0, 0, 0, 0, 0, 0, 0],
    ),
    chess.KNIGHT: (
        [-167, -89, -34, -49, 61, -97, -15, -107,
         -73, -41, 72, 36, 23, 62, 7, -17,
         -47, 60, 37, 65, 84, 129, 73, 44,
         -9, 17, 19, 53, 37, 69, 18, 22,
         -13, 4, 16, 13, 28, 19, 21, -8,
         -23, -9, 12, 10, 19, 17, 25, -16,
         -29, -53, -12, -3, -1, 18, -14, -19,
         -105, -21, -58, -33, -17, -28, -19, -23],
        [-58, -38, -13, -28, -31, -27, -63, -99,
         -25, -8, -25, -2, -9, -25, -24, -52,
         -24, -20, 10, 9, -1, -9, -19, -41,
         -17, 3, 22, 22, 22, 11, 8, -18,
         -18, -6, 16, 25, 16, 17, 4, -18,
         -23, -3, -1, 15, 10, -3, -20, -22,
         -42, -20, -10, -5, -2, -20, -23, -44,
         -29, -51, -23, -15, -22, -18, -50, -64],
    ),
    chess.BISHOP: (
        [-29, 4, -82, -37, -25, -42, 7, -8,
         -26, 16, -18, -13, 30, 59, 18, -47,
         -16, 37, 43, 40, 35, 50, 37, -2,
         -4, 5, 19, 50, 37, 37, 7, -2,
         -6, 13, 13, 26, 34, 12, 10, 4,
         0, 15, 15, 15, 14, 27, 18, 10,
         4, 15, 16, 0, 7, 21, 33, 1,
         -33, -3, -14, -21, -13, -12, -39, -21],
        [-14, -21, -11, -8, -7, -9, -17, -24,
         -8, -4, 7, -12, -3, -13, -4, -14,
         2, -8, 0, -1, -2, 6, 0, 4,
         -3, 9, 12, 9, 14, 10, 3, 2,
         -6, 3, 13, 19, 7, 10, -3, -9,
         -12, -3, 8, 10, 13, 3, -7, -15,
         -14, -18, -7, -1, 4, -9, -15, -27,
         -23, -9, -23, -5, -9, -16, -5, -17],
    ),
    chess.ROOK: (
        [32, 42, 32, 51, 63, 9, 31, 43,
         27, 32, 58, 62, 80, 67, 26, 44,
         -5, 19, 26, 36, 17, 45, 61, 16,
         -24, -11, 7, 26, 24, 35, -8, -20,
         -36, -26, -12, -1, 9, -7, 6, -23,
         -45, -25, -16, -17, 3, 0, -5, -33,
         -44, -16, -20, -9, -1, 11, -6, -71,
         -19, -13, 1, 17, 16, 7, -37, -26],
        [13, 10, 18, 15, 12, 12, 8, 5,
         11, 13, 13, 11, -3, 3, 8, 3,
         7, 7, 7, 5, 4, -3, -5, -3,
         4, 3, 13, 1, 2, 1, -1, 2,
         3, 5, 8, 4, -5, -6, -8, -11,
         -4, 0, -5, -1, -7, -12, -8, -16,
         -6, -6, 0, 2, -9, -9, -11, -3,
         -9, 2, 3, -1, -5, -13, 4, -20],
    ),
    chess.QUEEN: (
        [-28, 0, 29, 12, 59, 44, 43, 45,
         -24, -39, -5, 1, -16, 57, 28, 54,
         -13, -17, 7, 8, 29, 56, 47, 57,
         -27, -27, -16, -16, -1, 17, -2, 1,
         -9, -26, -9, -10, -2, -4, 3, -3,
         -14, 2, -11, -2, -5, 2, 14, 5,
         -35, -8, 11, 2, 8, 15, -3, 1,
         -1, -18, -9, 10, -15, -25, -31, -50],
        [-9, 22, 22, 27, 27, 19, 10, 20,
         -17, 20, 32, 41, 58, 25, 30, 0,
         -20, 6, 9, 49, 47, 35, 19, 9,
         3, 22, 24, 45, 57, 40, 57, 36,
         -18, 28, 19, 47, 31, 34, 39, 23,
         -16, -27, 15, 6, 9, 17, 10, 5,
         -22, -23, -30, -16, -16, -23, -36, -32,
         -33, -28, -22, -43, -5, -32, -20, -41],
    ),
    chess.KING: (
        [-65, 23, 16, -15, -56, -34, 2, 13,
         29, -1, -20, -7, -8, -4, -38, -29,
         -9, 24, 2, -16, -20, 6, 22, -22,
         -17, -20, -12, -27, -30, -25, -14, -36,
         -49, -1, -27, -39, -46, -44, -33, -51,
         -14, -14, -22, -46, -44, -30, -15, -27,
         1, 7, -8, -64, -43, -16, 9, 8,
         -15, 36, 12, -54, 8, -28, 24, 14],
        [-74, -35, -18, -18, -11, 15, 4, -17,
         -12, 17, 14, 17, 17, 38, 23, 11,
         10, 17, 23, 15, 20, 45, 44, 13,
         -8, 22, 24, 27, 26, 33, 26, 3,
         -18, -4, 21, 24, 27, 23, 9, -11,
         -19, -3, 11, 21, 23, 16, 7, -9,
         -27, -11, 4, 13, 14, 4, -5, -17,
         -53, -34, -21, -11, -28, -14, -24, -43],
    ),
}


def _to_square_indexed(published: list[int]) -> list[int]:
    """Flip a table given a8->h1 (top row first) into python-chess's a1->h8 order."""
    rows = [published[r * 8 : r * 8 + 8] for r in range(8)]
    rows.reverse()
    return [value for row in rows for value in row]


def _combined(published: list[int], value: int, mirror: bool) -> list[int]:
    """One lookup per piece per square, with the material value already folded in.

    `mirror` builds the Black copy, so the evaluation loop never pays for a
    `square_mirror` call or a sign flip: it reads a different table instead.
    """
    table = _to_square_indexed(published)
    if mirror:
        table = [table[chess.square_mirror(square)] for square in chess.SQUARES]
    return [value + entry for entry in table]


# Indexed [piece_type][square]; entry 0 is padding so piece types index directly.
_EMPTY: list[int] = [0] * 64
MG_WHITE = [_EMPTY] + [_combined(_RAW_PST[pt][0], MG_VALUE[pt], False) for pt in chess.PIECE_TYPES]
EG_WHITE = [_EMPTY] + [_combined(_RAW_PST[pt][1], EG_VALUE[pt], False) for pt in chess.PIECE_TYPES]
MG_BLACK = [_EMPTY] + [_combined(_RAW_PST[pt][0], MG_VALUE[pt], True) for pt in chess.PIECE_TYPES]
EG_BLACK = [_EMPTY] + [_combined(_RAW_PST[pt][1], EG_VALUE[pt], True) for pt in chess.PIECE_TYPES]

# --- structural terms, all (middlegame, endgame) pairs in centipawns ---------

BISHOP_PAIR = (24, 46)
ROOK_OPEN_FILE = (26, 12)
ROOK_SEMI_OPEN_FILE = (11, 6)
DOUBLED_PAWN = (-9, -22)
ISOLATED_PAWN = (-13, -16)
PHALANX_PAWN = (6, 4)
# By the pawn's rank counted from its own side, so index 6 is one step from
# queening. A passer is worth a little while pieces are on and a great deal once
# there is nothing left to stop it, which is what the taper is for.
PASSED_PAWN_MG = (0, 4, 6, 14, 30, 58, 96, 0)
PASSED_PAWN_EG = (0, 12, 20, 36, 64, 106, 170, 0)
# Per pawn standing in the two ranks in front of the king, up to three.
KING_SHIELD = 9
# Mobility, per square a piece attacks that its own men do not occupy, indexed
# by piece type. Rooks and queens gain as the board empties.
MOBILITY_MG = (0, 0, 4, 5, 2, 1, 0)
MOBILITY_EG = (0, 0, 4, 5, 4, 3, 0)
# King safety. Every square in the ring around a king that an enemy piece
# attacks counts, weighted by what attacks it, and the total is read off a table
# that grows faster than linearly -- two attackers are far more than twice one.
# It only applies from the second attacker on, because one piece pointing at a
# king is not an attack.
KING_ATTACK_WEIGHT = (0, 0, 2, 2, 3, 5, 0)
KING_DANGER = tuple(min(300, units * units // 6) for units in range(96))
MAX_DANGER_INDEX = len(KING_DANGER) - 1
# Having the move is worth something, and saying so stops the evaluation from
# swinging by a whole tempo between odd and even plies.
TEMPO = 10

FILE_BB = list(chess.BB_FILES)
ADJACENT_FILES = [
    (FILE_BB[file - 1] if file > 0 else 0) | (FILE_BB[file + 1] if file < 7 else 0)
    for file in range(8)
]
KING_FILES = [FILE_BB[file] | ADJACENT_FILES[file] for file in range(8)]


def _ahead_of(colour: chess.Color, square: int, files: int) -> int:
    """The squares of `files` lying strictly ahead of `square`, from colour's view."""
    rank = chess.square_rank(square)
    ranks = range(rank + 1, 8) if colour == chess.WHITE else range(0, rank)
    span = 0
    for index in ranks:
        span |= chess.BB_RANKS[index]
    return span & files


def _shield_of(colour: chess.Color, square: int) -> int:
    """The two ranks in front of a king, across its own file and its neighbours."""
    rank = chess.square_rank(square)
    ranks = (rank + 1, rank + 2) if colour == chess.WHITE else (rank - 1, rank - 2)
    mask = 0
    for index in ranks:
        if 0 <= index <= 7:
            mask |= chess.BB_RANKS[index]
    return mask & KING_FILES[chess.square_file(square)]


# Indexed [colour][square]. chess.BLACK is False and chess.WHITE is True, so
# building the outer list in that order lets a colour index it directly.
FORWARD_FILE = [
    [_ahead_of(colour, square, FILE_BB[chess.square_file(square)]) for square in chess.SQUARES]
    for colour in (chess.BLACK, chess.WHITE)
]
PASSED_MASK = [
    [_ahead_of(colour, square, KING_FILES[chess.square_file(square)]) for square in chess.SQUARES]
    for colour in (chess.BLACK, chess.WHITE)
]
KING_SHIELD_MASK = [
    [_shield_of(colour, square) for square in chess.SQUARES]
    for colour in (chess.BLACK, chess.WHITE)
]

MATE_SCORE = 100_000
DRAW_SCORE = 0
# What a piece is worth to an exchange on a single square, in round numbers.
# Kept apart from the tapered tables because the swap-off arithmetic in `_see`
# wants one number per piece type, not two.
SEE_VALUE = (0, 100, 320, 330, 500, 900, 20_000)


def _piece_attacks(piece_type: int, square: int, occupied: int) -> int:
    """The squares a piece of this type on this square attacks, blockers included.

    python-chess exposes the same magic tables its own move generator uses, and
    going straight to them beats `Board.attacks_mask`, which has to work out the
    piece type from the bitboards first. In the evaluation loop we already know.
    """
    if piece_type == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[square]
    diagonal = chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
    if piece_type == chess.BISHOP:
        return diagonal
    straight = (
        chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
        | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
    )
    if piece_type == chess.ROOK:
        return straight
    return diagonal | straight


# Pawn structure depends only on where the pawns are, and pawns move rarely, so
# the same two bitboards come back again and again inside one search. Caching
# the answer takes the whole loop off the hot path for most nodes.
_pawn_cache: dict[tuple[int, int], tuple[int, int]] = {}
PAWN_CACHE_LIMIT = 100_000


def _pawn_structure(white_pawns: int, black_pawns: int) -> tuple[int, int]:
    """Doubled, isolated, connected and passed pawns, as a (middlegame, endgame) pair.

    White-relative: a positive number favours White. Everything here is a
    property of the pawn skeleton alone, which is what makes it cacheable.
    """
    cached = _pawn_cache.get((white_pawns, black_pawns))
    if cached is not None:
        return cached

    mg = 0
    eg = 0
    for colour, own, enemy, sign in (
        (chess.WHITE, white_pawns, black_pawns, 1),
        (chess.BLACK, black_pawns, white_pawns, -1),
    ):
        forward = FORWARD_FILE[colour]
        passed_mask = PASSED_MASK[colour]
        remaining = own
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            file = square & 7
            rank = square >> 3
            relative = rank if colour == chess.WHITE else 7 - rank
            ahead = forward[square]
            if not own & ADJACENT_FILES[file]:
                mg += sign * ISOLATED_PAWN[0]
                eg += sign * ISOLATED_PAWN[1]
            elif own & ADJACENT_FILES[file] & chess.BB_RANKS[rank]:
                mg += sign * PHALANX_PAWN[0]
                eg += sign * PHALANX_PAWN[1]
            if own & ahead:
                mg += sign * DOUBLED_PAWN[0]
                eg += sign * DOUBLED_PAWN[1]
            elif not enemy & passed_mask[square]:
                mg += sign * PASSED_PAWN_MG[relative]
                eg += sign * PASSED_PAWN_EG[relative]

    if len(_pawn_cache) > PAWN_CACHE_LIMIT:
        _pawn_cache.clear()
    _pawn_cache[(white_pawns, black_pawns)] = (mg, eg)
    return mg, eg


def _pawn_ending_score(board: chess.Board) -> int:
    """Score a solved king-and-pawn-versus-king position, side to move first.

    The bitbase answers won or drawn outright, so a drawn one is worth exactly
    nothing however far advanced the pawn is -- that is the whole point of
    consulting it, and it is what stops us trading into a dead ending. A won one
    still needs a gradient to walk down, so it is scored by how close the pawn is
    to queening with its own king escorting it.
    """
    winner = endgame.probe(board)
    if winner is None:
        return DRAW_SCORE
    pawn = chess.msb(board.pawns)
    rank = chess.square_rank(pawn) if winner == chess.WHITE else 7 - chess.square_rank(pawn)
    queening = chess.square(chess.square_file(pawn), 7 if winner == chess.WHITE else 0)
    king = chess.msb(board.kings & board.occupied_co[winner])
    score = (
        PAWN_ENDING_WIN
        + PAWN_ENDING_PUSH * rank
        - PAWN_ENDING_ESCORT * chess.square_distance(king, queening)
    )
    return score if board.turn == winner else -score


def _bare_king_side(board: chess.Board) -> chess.Color | None:
    """The side down to a bare king, when the other side has a rook or queen to mate it.

    Restricted to a rook or queen on purpose. Those mate by driving the king to
    any edge, which is what `_lone_king_score` rewards. Bishop-and-knight needs a
    corner of one particular colour and two bishops need their own net, so the
    ordinary evaluation keeps those; a side with only pawns has to promote first,
    and dropping its piece-square tables would take away the reason to.
    """
    for loser in (chess.WHITE, chess.BLACK):
        bare = board.occupied_co[loser] == (board.kings & board.occupied_co[loser])
        can_mate = board.occupied_co[not loser] & (board.queens | board.rooks)
        if bare and can_mate:
            return loser
    return None


def _lone_king_score(board: chess.Board, loser: chess.Color) -> int:
    """Score a bare-king position from White's point of view.

    The piece-square tables are dropped here deliberately. They are tuned for
    real positions, and their gradients are far larger than any king-herding term
    worth adding, so with a bare king about they simply drown it out: the winner
    shuffles its queen between "good" squares while the defender walks free.
    What is left is material, so we still never give a piece away, plus the two
    terms that actually end the game -- push the lone king to the edge, and walk
    our own king up to it. King distance is measured the long way round the
    board rather than diagonally, which grades the approach far more finely.
    """
    material = 0
    for piece in board.piece_map().values():
        sign = 1 if piece.color == chess.WHITE else -1
        material += sign * EG_VALUE[piece.piece_type]
    weak_king = chess.msb(board.kings & board.occupied_co[loser])
    strong_king = chess.msb(board.kings & board.occupied_co[not loser])
    drive = EDGE_WEIGHT * CENTRE_DISTANCE[weak_king] + APPROACH_WEIGHT * (
        14 - chess.square_manhattan_distance(strong_king, weak_king)
    )
    return material - drive if loser == chess.WHITE else material + drive


def _ordinary_score(board: chess.Board) -> int:
    """The evaluation proper, for a position with no solved endgame shortcut.

    This is the readable definition of what a position is worth, and it stays
    the authority even when the compiled twin in `fasteval` is what actually
    runs: that module is checked against this one at import.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    occupied = board.occupied
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens

    # Material and piece-square tables, plus the phase count that blends them.
    mg = 0
    eg = 0
    phase = 0
    for piece_type, pieces in (
        (chess.PAWN, pawns),
        (chess.KNIGHT, knights),
        (chess.BISHOP, bishops),
        (chess.ROOK, rooks),
        (chess.QUEEN, queens),
        (chess.KING, board.kings),
    ):
        weight = PHASE_WEIGHT[piece_type]
        mg_white = MG_WHITE[piece_type]
        eg_white = EG_WHITE[piece_type]
        mg_black = MG_BLACK[piece_type]
        eg_black = EG_BLACK[piece_type]
        remaining = pieces & white
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            mg += mg_white[square]
            eg += eg_white[square]
            phase += weight
        remaining = pieces & black
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            mg -= mg_black[square]
            eg -= eg_black[square]
            phase += weight

    if (bishops & white).bit_count() >= 2:
        mg += BISHOP_PAIR[0]
        eg += BISHOP_PAIR[1]
    if (bishops & black).bit_count() >= 2:
        mg -= BISHOP_PAIR[0]
        eg -= BISHOP_PAIR[1]

    pawn_mg, pawn_eg = _pawn_structure(pawns & white, pawns & black)
    mg += pawn_mg
    eg += pawn_eg

    # Rooks want files their own pawns are not standing on, and ideally files
    # nobody's pawns are standing on.
    for pieces, own_pawns, sign in (
        (rooks & white, pawns & white, 1),
        (rooks & black, pawns & black, -1),
    ):
        remaining = pieces
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            file_mask = FILE_BB[square & 7]
            if not pawns & file_mask:
                mg += sign * ROOK_OPEN_FILE[0]
                eg += sign * ROOK_OPEN_FILE[1]
            elif not own_pawns & file_mask:
                mg += sign * ROOK_SEMI_OPEN_FILE[0]
                eg += sign * ROOK_SEMI_OPEN_FILE[1]

    # Mobility and king safety share one pass: the attack set a piece needs for
    # "how many squares do I have" is the same set that answers "am I pointing at
    # the enemy king", so generating it twice would be paying twice.
    white_king = (board.kings & white).bit_length() - 1
    black_king = (board.kings & black).bit_length() - 1
    white_zone = chess.BB_KING_ATTACKS[black_king] | chess.BB_SQUARES[black_king]
    black_zone = chess.BB_KING_ATTACKS[white_king] | chess.BB_SQUARES[white_king]
    white_units = 0
    white_attackers = 0
    black_units = 0
    black_attackers = 0
    for piece_type, pieces in (
        (chess.KNIGHT, knights),
        (chess.BISHOP, bishops),
        (chess.ROOK, rooks),
        (chess.QUEEN, queens),
    ):
        mobility_mg = MOBILITY_MG[piece_type]
        mobility_eg = MOBILITY_EG[piece_type]
        attack_weight = KING_ATTACK_WEIGHT[piece_type]
        remaining = pieces & white
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            attacks = _piece_attacks(piece_type, square, occupied)
            free = (attacks & ~white).bit_count()
            mg += mobility_mg * free
            eg += mobility_eg * free
            near_king = (attacks & white_zone).bit_count()
            if near_king:
                white_units += attack_weight * near_king
                white_attackers += 1
        remaining = pieces & black
        while remaining:
            square = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            attacks = _piece_attacks(piece_type, square, occupied)
            free = (attacks & ~black).bit_count()
            mg -= mobility_mg * free
            eg -= mobility_eg * free
            near_king = (attacks & black_zone).bit_count()
            if near_king:
                black_units += attack_weight * near_king
                black_attackers += 1

    if white_attackers >= 2:
        mg += KING_DANGER[min(white_units, MAX_DANGER_INDEX)]
    if black_attackers >= 2:
        mg -= KING_DANGER[min(black_units, MAX_DANGER_INDEX)]
    white_shield = (pawns & white & KING_SHIELD_MASK[chess.WHITE][white_king]).bit_count()
    black_shield = (pawns & black & KING_SHIELD_MASK[chess.BLACK][black_king]).bit_count()
    mg += KING_SHIELD * min(3, white_shield)
    mg -= KING_SHIELD * min(3, black_shield)

    if phase > MAX_PHASE:
        phase = MAX_PHASE
    # Truncate towards zero rather than floor: flooring makes the score of a
    # position and the score of its colour-swapped mirror differ by a centipawn,
    # which is a bias towards one colour and nothing else.
    tapered = int((mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE)
    return (tapered if board.turn == chess.WHITE else -tapered) + TEMPO


if _evaluator is not None:
    try:
        _evaluator.install(
            MG_WHITE, EG_WHITE, MG_BLACK, EG_BLACK, PHASE_WEIGHT,
            FORWARD_FILE, PASSED_MASK, KING_SHIELD_MASK,
            KING_DANGER, MOBILITY_MG, MOBILITY_EG, KING_ATTACK_WEIGHT,
        )
    except Exception as _exc:  # a compile failure is not worth a lost game
        _evaluator = None
        print(f"fasteval would not compile, using the Python evaluation: {_exc!r}")


def _compiled_score(board: chess.Board) -> int:
    """The same number as `_ordinary_score`, computed by the compiled version."""
    return int(
        _evaluator.score(
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
            board.turn,
        )
    )


def _compiled_agrees(positions: int = 1200) -> bool:
    """Walk random games and insist the two evaluations return the same number.

    numba is the one thing here whose behaviour we cannot fully predict on a
    machine we never see, and a silently different evaluation would play bad
    moves for a whole game without ever looking broken. Checking costs a
    fraction of a second of the import budget, and the cost of being wrong is
    the game.
    """
    generator = random.Random(20260910)
    board = chess.Board()
    for _ in range(positions):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
            continue
        board.push(generator.choice(moves))
        if endgame.is_pawn_ending(board) or _bare_king_side(board) is not None:
            continue
        if _compiled_score(board) != _ordinary_score(board):
            return False
    return True


_started_at = time.monotonic()
_USE_COMPILED = _evaluator is not None and _compiled_agrees()
_score_position = _compiled_score if _USE_COMPILED else _ordinary_score
print(
    f"evaluation: {'compiled' if _USE_COMPILED else 'PYTHON FALLBACK'}, "
    f"ready in {time.monotonic() - _started_at:.1f}s"
)


def evaluate(board: chess.Board) -> int:
    """Score `board` from the perspective of the side to move (negamax convention)."""
    if endgame.is_pawn_ending(board):
        return _pawn_ending_score(board)
    loser = _bare_king_side(board)
    if loser is not None:
        white_relative = _lone_king_score(board, loser)
        return white_relative if board.turn == chess.WHITE else -white_relative
    return _score_position(board)


# ---------------------------------------------------------------------------
# Static exchange evaluation.
#
# "If I capture on this square and both sides keep recapturing with their
# cheapest attacker, what do I end up with?" The search uses the answer twice:
# to keep quiescence from playing out captures that only lose material, and to
# push obviously losing captures to the back of the move order. It ignores pins,
# which is the usual trade -- a pinned recapture makes it slightly pessimistic,
# and the real search corrects that a ply later.
# ---------------------------------------------------------------------------


def _attackers_to(board: chess.Board, square: int, occupied: int) -> int:
    """Every piece of either colour attacking `square`, given this occupancy.

    Occupancy is a parameter rather than the board's own, because the swap-off
    below removes each capturer in turn and has to see the sliders behind them.
    """
    rank_pieces = chess.BB_RANK_MASKS[square] & occupied
    file_pieces = chess.BB_FILE_MASKS[square] & occupied
    diagonal_pieces = chess.BB_DIAG_MASKS[square] & occupied
    straight_movers = board.queens | board.rooks
    diagonal_movers = board.queens | board.bishops
    pawns = board.pawns
    white_pawns = pawns & board.occupied_co[chess.WHITE]
    black_pawns = pawns & board.occupied_co[chess.BLACK]
    return occupied & (
        (chess.BB_RANK_ATTACKS[square][rank_pieces] & straight_movers)
        | (chess.BB_FILE_ATTACKS[square][file_pieces] & straight_movers)
        | (chess.BB_DIAG_ATTACKS[square][diagonal_pieces] & diagonal_movers)
        | (chess.BB_KNIGHT_ATTACKS[square] & board.knights)
        | (chess.BB_KING_ATTACKS[square] & board.kings)
        | (chess.BB_PAWN_ATTACKS[chess.BLACK][square] & white_pawns)
        | (chess.BB_PAWN_ATTACKS[chess.WHITE][square] & black_pawns)
    )


def _see(board: chess.Board, move: chess.Move) -> int:
    """Centipawns the side to move nets from this capture after all recaptures."""
    square = move.to_square
    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        captured = SEE_VALUE[chess.PAWN]
        behind = square - 8 if board.turn == chess.WHITE else square + 8
        occupied &= ~chess.BB_SQUARES[behind]
    else:
        victim = board.piece_type_at(square)
        captured = SEE_VALUE[victim] if victim is not None else 0
    mover = board.piece_type_at(move.from_square)
    if mover is None:
        return 0
    standing = SEE_VALUE[mover]  # what is left sitting on the square to be taken
    if move.promotion is not None:
        captured += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        standing = SEE_VALUE[move.promotion]

    # Walk the recaptures forward, cheapest attacker first, taking each capturer
    # off the occupancy so the slider behind it comes into view. `taken` ends up
    # holding what each recapture in turn would win.
    taken: list[int] = []
    by_type = (0, board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    side = not board.turn
    attackers = _attackers_to(board, square, occupied) & occupied
    while True:
        available = attackers & board.occupied_co[side]
        if not available:
            break
        for piece_type in chess.PIECE_TYPES:
            subset = available & by_type[piece_type]
            if subset:
                break
        else:
            break
        # A king cannot recapture onto a square the other side still attacks.
        if piece_type == chess.KING and attackers & board.occupied_co[not side]:
            break
        taken.append(standing)
        standing = SEE_VALUE[piece_type]
        occupied &= ~(subset & -subset)
        attackers = _attackers_to(board, square, occupied) & occupied
        side = not side

    # Now unwind. Neither side has to keep capturing, so from the last recapture
    # backwards each one is worth what it wins less what the reply is worth, or
    # nothing at all if that comes out negative and the side simply stops. The
    # opening capture is the one move that is already committed, so it alone does
    # not get to decline.
    reply = 0
    for winnings in reversed(taken):
        reply = max(0, winnings - reply)
    return captured - reply


# ---------------------------------------------------------------------------
# Search state.
#
# These persist across our own moves within one game (a fresh process starts per
# game, per the contract), which is what makes a transposition table and a
# history heuristic worth having: work from an earlier move keeps paying off
# later in the same game.
# ---------------------------------------------------------------------------

EXACT, LOWER, UPPER = 0, 1, 2
# (depth, score, flag, best move). A plain tuple rather than a dataclass: at a
# few hundred thousand live entries the allocation and the attribute lookups
# both show up in a profile.
TTEntry = tuple[int, int, int, "chess.Move | None"]

transposition_table: dict[int, TTEntry] = {}
# Each entry is an int key and a four-tuple, so a few hundred megabytes at most,
# well inside the 2 GB the container gives us. When it fills, throw it away
# rather than trying to be clever about which half to keep.
TT_LIMIT = 800_000

killer_moves: dict[int, list[chess.Move]] = {}
history_heuristic: dict[tuple[bool, int, int], int] = {}
HISTORY_CAP = 1 << 14
# The reply that refuted the opponent's last move, keyed by that move. Quiet
# refutations tend to work again when the same move is played again elsewhere.
counter_moves: dict[tuple[bool, int, int], chess.Move] = {}
# Position key -> how many times it has stood on the board, counting both the
# real game so far and the line the search is currently walking. Reaching a key
# already in here means the line repeats, and a repetition is a draw, which a
# board built fresh from one FEN would otherwise be blind to.
game_history: dict[int, int] = {}

MAX_DEPTH = 64
MAX_PLY = 96
# No evaluation can reach this, so a score above it is a forced mate and nothing
# else. The margin is generous: it only has to clear the deepest ply we reach.
MATE_BOUND = MATE_SCORE - 1000

# Pruning and reduction knobs.
RFP_MARGIN = 85
DELTA_MARGIN = 120
# Indexed by depth. A quiet move at depth one or two that cannot get within this
# of alpha even granting it a free tempo is not going to raise alpha.
FUTILITY_MARGIN = (0, 150, 300)
ASPIRATION_WINDOW = 25
# How far back a late, quiet move gets reduced, by (depth, move number). The
# logarithms are the standard shape: reduce more the deeper we are and the
# further down the move list we have got.
LMR = [
    [
        0 if depth < 3 or number < 3 else int(0.75 + math.log(depth) * math.log(number) / 2.25)
        for number in range(64)
    ]
    for depth in range(64)
]


class TimeUp(Exception):
    pass


_nodes = 0


def _check_time(deadline: float) -> None:
    """Abort the search once the clock is up, checked often enough to be prompt.

    `time.monotonic` is cheap but not free, and at these node counts calling it
    every node is measurable, so sample it instead. 128 nodes is a few
    milliseconds of granularity against a budget measured in hundreds.
    """
    global _nodes
    _nodes += 1
    if not _nodes & 127 and time.monotonic() > deadline:
        raise TimeUp


def _key(board: chess.Board) -> int:
    """A position key: everything that makes two positions the same one.

    Cheaper than a Zobrist hash recomputed from the piece map, which is what
    `chess.polyglot` does and what dominated the profile before. Kings are not
    listed because `occupied` minus the other five boards is exactly where they
    are. Two positions can collide, as they can in any hashed table; a wrong
    score costs a bad move, never an illegal one, because every move we return
    came out of python-chess's own generator.
    """
    return hash(
        (
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.occupied_co[chess.WHITE],
            board.occupied,
            board.turn,
            board.castling_rights,
            board.ep_square,
        )
    )


def _score_to_tt(score: int, ply: int) -> int:
    """Re-base a mate score from "this far from the root" to "this far from here".

    A mate is scored as MATE_SCORE less the ply it was found at, which measures
    the distance from the root of the search that found it. The table outlives
    that search -- it is kept for the whole game -- so the same entry gets read
    back at some other ply, and a raw score would then claim a mate at the wrong
    distance, or claim one that is not there at all. Storing the distance from
    the node itself makes the entry independent of how the search reached it.
    """
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    """Undo `_score_to_tt`, turning a stored distance back into one from the root."""
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


def _has_non_pawn_material(board: chess.Board, colour: chess.Color) -> bool:
    """Whether `colour` has a piece other than pawns, which null-move pruning needs.

    A side down to king and pawns is the zugzwang case: there, being handed a free
    move is not a favour, and pretending otherwise prunes away real defences.
    """
    ours = board.occupied_co[colour]
    return bool(ours & (board.knights | board.bishops | board.rooks | board.queens))


# ---------------------------------------------------------------------------
# Move ordering.
#
# Alpha-beta only prunes well if strong moves are tried first, so before
# searching we sort candidates by a cheap guess at their value:
#   1. the move the transposition table already knows was best here
#   2. captures, ranked by MVV-LVA ("most valuable victim, least valuable
#      attacker"), with any capture that walks into a defended square for less
#      than it takes demoted below every quiet move instead
#   3. queen promotions
#   4. "killers": quiet moves that caused a beta cutoff at this same ply in a
#      sibling branch, then the quiet move that refuted this same reply last time
#   5. everything else by the history heuristic: quiet moves that have caused
#      cutoffs anywhere in the tree, tallied by (colour, from, to) so the ranking
#      survives across positions
# ---------------------------------------------------------------------------

TT_MOVE_SCORE = 3_000_000
GOOD_CAPTURE_SCORE = 2_000_000
PROMOTION_SCORE = 1_500_000
KILLER_SCORE = 1_400_000
COUNTER_SCORE = 1_300_000
BAD_CAPTURE_SCORE = -1_000_000


def _order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None,
    ply: int,
    counter: chess.Move | None,
) -> list[chess.Move]:
    killers = killer_moves.get(ply, ())
    turn = board.turn
    enemy = board.occupied_co[not turn]

    def score(move: chess.Move) -> int:
        if move == tt_move:
            return TT_MOVE_SCORE
        to_square = move.to_square
        if enemy & chess.BB_SQUARES[to_square] or board.is_en_passant(move):
            victim = board.piece_type_at(to_square)
            victim_value = SEE_VALUE[victim] if victim is not None else SEE_VALUE[chess.PAWN]
            attacker = board.piece_type_at(move.from_square)
            attacker_value = SEE_VALUE[attacker] if attacker is not None else 0
            base = victim_value * 16 - attacker_value
            if move.promotion is not None:
                base += SEE_VALUE[move.promotion] * 16
            elif victim_value < attacker_value and board.is_attacked_by(not turn, to_square):
                return BAD_CAPTURE_SCORE + base
            return GOOD_CAPTURE_SCORE + base
        if move.promotion is not None:
            return PROMOTION_SCORE + SEE_VALUE[move.promotion]
        if move in killers:
            return KILLER_SCORE - killers.index(move)
        if move == counter:
            return COUNTER_SCORE
        return history_heuristic.get((turn, move.from_square, move.to_square), 0)

    return sorted(moves, key=score, reverse=True)


def _counter_key(board: chess.Board) -> tuple[bool, int, int] | None:
    """Key the counter-move table by the reply we are answering, if there is one."""
    if not board.move_stack:
        return None
    previous = board.move_stack[-1]
    if previous.from_square == previous.to_square:
        return None  # a null move; nothing was actually played
    return (board.turn, previous.from_square, previous.to_square)


def _record_cutoff(
    board: chess.Board,
    move: chess.Move,
    ply: int,
    depth: int,
    quiets: list[chess.Move],
) -> None:
    """A quiet move caused a beta cutoff: promote it, and demote the ones that failed.

    Rewarding the winner alone leaves every untried quiet move sitting at the
    same score, so the ordering learns much faster if the moves that were tried
    first and did not cut off are pushed down by the same amount.
    """
    killers = killer_moves.setdefault(ply, [])
    if move not in killers:
        killers.insert(0, move)
        del killers[2:]
    counter_key = _counter_key(board)
    if counter_key is not None:
        counter_moves[counter_key] = move
    turn = board.turn
    bonus = depth * depth
    key = (turn, move.from_square, move.to_square)
    history_heuristic[key] = min(HISTORY_CAP, history_heuristic.get(key, 0) + bonus)
    for quiet in quiets:
        if quiet == move:
            continue
        failed = (turn, quiet.from_square, quiet.to_square)
        history_heuristic[failed] = max(-HISTORY_CAP, history_heuristic.get(failed, 0) - bonus)


def _tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and queen promotions: the moves quiescence is allowed to consider.

    Under-promotions are dropped. They matter perhaps once a game and generating
    them triples the promotion branch, which quiescence pays for at every leaf.
    """
    moves = [
        move
        for move in board.generate_legal_captures()
        if move.promotion is None or move.promotion == chess.QUEEN
    ]
    last_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
    promoting = board.pawns & board.occupied_co[board.turn] & last_rank
    if promoting:
        moves.extend(
            move
            for move in board.generate_legal_moves(promoting)
            if move.promotion == chess.QUEEN
            and not board.occupied & chess.BB_SQUARES[move.to_square]
        )
    return moves


QS_CHECK_DEPTH = -6


def quiescence(
    board: chess.Board, alpha: int, beta: int, ply: int, depth: int, deadline: float
) -> int:
    """Play out the captures at a leaf, so we never score a position mid-exchange.

    Three things keep it from exploding. Captures the static exchange evaluation
    says lose material are not searched at all. A capture that cannot bring the
    score near alpha even if the piece were free is skipped. And checks are
    answered in full for a few plies -- being in check makes standing pat a lie --
    but only a few, after which the static score has to do.
    """
    _check_time(deadline)
    if ply >= MAX_PLY:
        return evaluate(board)

    if board.is_check() and depth > QS_CHECK_DEPTH:
        moves = list(board.legal_moves)
        if not moves:
            # Counting the ply matters even down here: it is what makes a mate
            # found sooner score better than the same mate found later.
            return -MATE_SCORE + ply
        best = -MATE_SCORE - 1
        for move in _order_moves(board, moves, None, ply, None):
            board.push(move)
            try:
                score = -quiescence(board, -beta, -alpha, ply + 1, depth - 1, deadline)
            finally:
                board.pop()
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
        return best

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat
    best = stand_pat

    for move in _order_moves(board, _tactical_moves(board), None, ply, None):
        victim = board.piece_type_at(move.to_square)
        gain = SEE_VALUE[victim] if victim is not None else SEE_VALUE[chess.PAWN]
        if move.promotion is not None:
            gain += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        if stand_pat + gain + DELTA_MARGIN < alpha:
            continue
        if _see(board, move) < 0:
            continue
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1, depth - 1, deadline)
        finally:
            board.pop()
        if score > best:
            best = score
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    return best


def negamax(
    board: chess.Board,
    key: int,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    deadline: float,
    can_null: bool,
) -> int:
    _check_time(deadline)
    is_pv = beta - alpha > 1

    if ply:
        if (
            game_history.get(key, 0) >= 2
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            return DRAW_SCORE
        if ply >= MAX_PLY:
            return evaluate(board)
        # Mate distance pruning. A mate we could already deliver sooner than any
        # mate found down here makes this whole subtree irrelevant.
        if alpha < -MATE_SCORE + ply:
            alpha = -MATE_SCORE + ply
        if beta > MATE_SCORE - ply - 1:
            beta = MATE_SCORE - ply - 1
        if alpha >= beta:
            return alpha

    alpha_orig = alpha
    entry = transposition_table.get(key)
    tt_move = entry[3] if entry is not None else None
    if entry is not None and entry[0] >= depth and not is_pv:
        stored = _score_from_tt(entry[1], ply)
        flag = entry[2]
        if (
            flag == EXACT
            or (flag == LOWER and stored >= beta)
            or (flag == UPPER and stored <= alpha)
        ):
            return stored

    if depth <= 0:
        return quiescence(board, alpha, beta, ply, 0, deadline)

    # No table move here means the ordering at this node is a guess, and a deep
    # search on a guessed order is mostly wasted. Search one ply shallower; the
    # entry that leaves behind makes the re-search that follows much cheaper.
    if tt_move is None and depth >= 4:
        depth -= 1

    static = 0
    in_check = board.is_check()
    if in_check:
        # Check extension: a forced sequence is exactly where a fixed depth cuts
        # the line off halfway and reports a score the position never reaches.
        depth += 1
    else:
        static = evaluate(board)
        if not is_pv and abs(beta) < MATE_BOUND:
            # Reverse futility: so far ahead that even losing a chunk per ply of
            # what is left cannot bring us back under beta.
            if depth <= 6 and static - RFP_MARGIN * depth >= beta:
                return static
            # Null move: hand the opponent a free move. If they still cannot
            # claw back to beta, a real move will not do it either. Skipped when
            # we are down to king and pawns, where passing is genuinely bad and
            # the shortcut would prune the defence that saves us. The null
            # position is deliberately kept out of `game_history`: the game never
            # stood there, and counting it would invent repetitions.
            if (
                can_null
                and depth >= 3
                and static >= beta
                and _has_non_pawn_material(board, board.turn)
            ):
                reduction = 2 + depth // 6
                board.push(chess.Move.null())
                try:
                    score = -negamax(
                        board,
                        _key(board),
                        depth - 1 - reduction,
                        -beta,
                        -beta + 1,
                        ply + 1,
                        deadline,
                        False,
                    )
                finally:
                    board.pop()
                if score >= beta:
                    return beta if score >= MATE_BOUND else score

    moves = list(board.legal_moves)
    if not moves:
        return -MATE_SCORE + ply if in_check else DRAW_SCORE

    counter_key = _counter_key(board)
    counter = counter_moves.get(counter_key) if counter_key is not None else None
    ordered = _order_moves(board, moves, tt_move, ply, counter)

    best_score = -MATE_SCORE - 1
    best_move = ordered[0]
    quiets: list[chess.Move] = []
    for number, move in enumerate(ordered):
        capture = board.is_capture(move)
        board.push(move)
        child_key = _key(board)
        gives_check = board.is_check()
        game_history[child_key] = game_history.get(child_key, 0) + 1
        try:
            # Futility: at the last ply or two before quiescence, a quiet move
            # from a position already this far below alpha has no way to get
            # back, and searching it only confirms that.
            if (
                number
                and not is_pv
                and not in_check
                and not gives_check
                and not capture
                and depth <= 2
                and move.promotion is None
                and best_score > -MATE_BOUND
                and static + FUTILITY_MARGIN[depth] <= alpha
            ):
                continue
            # Late move reductions. Once the ordering has been wrong about the
            # first few moves and we are still here, the rest are unlikely to be
            # best, so search them shallower and only pay full price for the ones
            # that come back above alpha anyway.
            reduction = 0
            if not capture and not in_check and not gives_check and move.promotion is None:
                reduction = LMR[depth if depth < 64 else 63][number if number < 64 else 63]
                if is_pv and reduction:
                    reduction -= 1
                if reduction > depth - 2:
                    reduction = depth - 2
                if reduction < 0:
                    reduction = 0
            if not number:
                score = -negamax(
                    board, child_key, depth - 1, -beta, -alpha, ply + 1, deadline, True
                )
            else:
                score = -negamax(
                    board,
                    child_key,
                    depth - 1 - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    deadline,
                    True,
                )
                if reduction and score > alpha:
                    score = -negamax(
                        board, child_key, depth - 1, -alpha - 1, -alpha, ply + 1, deadline, True
                    )
                if alpha < score < beta:
                    score = -negamax(
                        board, child_key, depth - 1, -beta, -alpha, ply + 1, deadline, True
                    )
        finally:
            game_history[child_key] -= 1
            if not game_history[child_key]:
                del game_history[child_key]
            board.pop()

        if not capture:
            quiets.append(move)
        if score > best_score:
            best_score = score
            best_move = move
            if best_score > alpha:
                alpha = best_score
                if alpha >= beta:
                    if not capture:
                        _record_cutoff(board, move, ply, depth, quiets)
                    break

    if best_score <= alpha_orig:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    else:
        flag = EXACT
    if len(transposition_table) > TT_LIMIT:
        transposition_table.clear()
    transposition_table[key] = (depth, _score_to_tt(int(best_score), ply), flag, best_move)
    return best_score


def _root_search(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float,
    root_moves: list[chess.Move],
    scores: dict[chess.Move, int],
) -> tuple[int, chess.Move | None]:
    """One iteration over the root moves, returning the best it managed to prove.

    If the clock runs out partway through, the result is still usable as long as
    some move beat the window we came in with -- that move has been searched to
    this depth and beaten everything before it. A fail-low has proved nothing, so
    it re-raises and the caller keeps the previous iteration's answer.
    """
    entering = alpha
    best_score = -MATE_SCORE - 1
    best_move: chess.Move | None = None
    try:
        for number, move in enumerate(root_moves):
            board.push(move)
            child_key = _key(board)
            game_history[child_key] = game_history.get(child_key, 0) + 1
            try:
                if not number:
                    score = -negamax(board, child_key, depth - 1, -beta, -alpha, 1, deadline, True)
                else:
                    score = -negamax(
                        board, child_key, depth - 1, -alpha - 1, -alpha, 1, deadline, True
                    )
                    if alpha < score < beta:
                        score = -negamax(
                            board, child_key, depth - 1, -beta, -alpha, 1, deadline, True
                        )
            finally:
                game_history[child_key] -= 1
                if not game_history[child_key]:
                    del game_history[child_key]
                board.pop()
            scores[move] = int(score)
            if score > best_score:
                best_score = int(score)
                best_move = move
                if best_score > alpha:
                    alpha = best_score
                    if alpha >= beta:
                        break
    except TimeUp:
        if best_move is None or best_score <= entering:
            raise
    return best_score, best_move


# ---------------------------------------------------------------------------
# Time management.
#
# The contract gives 120 s plus half a second a move, and `time_left_ms` is what
# is left before this move -- the increment lands after. We are never told what
# the increment is, so we measure it: the clock we get handed back, minus what
# we left on it, is what was added. Until there is a sample we assume nothing,
# which only makes the first move of a game slightly cautious.
#
# Two limits come out of it. The soft one decides whether to start another
# iteration of deepening, since the next one costs several times the last. The
# hard one aborts mid-iteration, and only fires when the soft limit misjudged.
# ---------------------------------------------------------------------------

SAFETY_MARGIN_MS = 300
MIN_BUDGET_MS = 20
# What counts as low on time, as a multiple of the increment, and the fraction of
# the increment to spend once we are.
LOW_CLOCK_MULTIPLE = 10
LOW_CLOCK_SPEND = 0.6
MAX_INCREMENT_MS = 5_000

_previous_left_ms: int | None = None
_previous_spent_ms = 0.0
_increment_ms = 0.0
_increment_seen = False


def _observe_clock(time_left_ms: int) -> None:
    """Work out the increment from how the clock moved across our last move."""
    global _increment_ms, _increment_seen
    if _previous_left_ms is None:
        return
    sample = time_left_ms - (_previous_left_ms - _previous_spent_ms)
    sample = max(0.0, min(float(MAX_INCREMENT_MS), sample))
    # Overhead outside our own measurement can only eat into the sample, never
    # inflate it, so the smallest one seen is the honest estimate.
    _increment_ms = sample if not _increment_seen else min(_increment_ms, sample)
    _increment_seen = True


def _limits(time_left_ms: int, fullmove_number: int) -> tuple[float, float]:
    """Seconds to aim for, and seconds not to exceed, on this move."""
    usable = max(0.0, time_left_ms - SAFETY_MARGIN_MS)
    # Rated games here run long -- well past move sixty -- so the pot is divided
    # by a count that only tightens slowly, and the increment is spendable on top
    # of it because it arrives whatever we do.
    moves_left = max(20, 40 - fullmove_number // 2)
    soft = usable / moves_left + _increment_ms * 0.75
    soft = min(soft, usable * 0.35)
    hard = min(soft * 2.6, usable * 0.55)
    # Running low, spend under the increment so the clock climbs back instead of
    # settling just above zero. Without this the budget converges on "spend what
    # arrives", which parks us a few hundred milliseconds from a flag for the
    # rest of a long game and leaves nothing to absorb one slow move.
    if _increment_ms and time_left_ms < LOW_CLOCK_MULTIPLE * _increment_ms:
        soft = min(soft, _increment_ms * LOW_CLOCK_SPEND)
        hard = min(hard, _increment_ms * LOW_CLOCK_SPEND * 1.5)
    return max(soft, MIN_BUDGET_MS) / 1000, max(hard, MIN_BUDGET_MS) / 1000


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation. See module docstring for the contract."""
    global _previous_left_ms, _previous_spent_ms

    started = time.monotonic()
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # The platform should never ask us to move with none available, but
        # returning something here is still better than raising.
        return chess.Move.null().uci()
    fallback = legal_moves[0]

    _observe_clock(time_left_ms)
    _previous_left_ms = time_left_ms
    _previous_spent_ms = 0.0
    if len(legal_moves) == 1:
        _previous_spent_ms = (time.monotonic() - started) * 1000
        return fallback.uci()

    root_key = _key(board)
    game_history[root_key] = game_history.get(root_key, 0) + 1
    killer_moves.clear()
    # Age the history rather than dropping it: what worked last move is still a
    # decent guess, but it should not outweigh what is working now.
    for history_key in history_heuristic:
        history_heuristic[history_key] //= 2

    soft, hard = _limits(time_left_ms, board.fullmove_number)
    deadline = started + hard
    iteration_started = started

    best_move = fallback
    scores: dict[chess.Move, int] = {}
    root_moves = list(legal_moves)
    score = 0
    depth = 0
    stable = 0
    try:
        for depth in range(1, MAX_DEPTH + 1):
            previous_best = best_move
            window = ASPIRATION_WINDOW
            if depth >= 4:
                alpha, beta = score - window, score + window
            else:
                alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
            while True:
                value, move = _root_search(board, depth, alpha, beta, deadline, root_moves, scores)
                if value <= alpha and alpha > -MATE_SCORE - 1:
                    # Fail low: nothing beat the window, so widen downwards and
                    # try again. Nothing has been proved, so no move is adopted.
                    window *= 3
                    alpha = max(-MATE_SCORE - 1, value - window)
                    continue
                if move is not None:
                    best_move = move
                if value >= beta and beta < MATE_SCORE + 1:
                    window *= 3
                    beta = min(MATE_SCORE + 1, value + window)
                    continue
                score = value
                break

            root_moves.sort(key=lambda candidate: scores.get(candidate, -MATE_SCORE), reverse=True)
            stable = stable + 1 if best_move == previous_best else 1
            if abs(score) >= MATE_BOUND:
                break  # a forced mate; deeper search cannot improve on it
            now = time.monotonic()
            elapsed = now - started
            # Only start another iteration if it looks like finishing near the
            # budget. Each one costs a small multiple of the last, so the last
            # one's duration is the estimate to use -- testing elapsed on its own
            # overshoots by the whole width of the next iteration, or, corrected
            # for that, stops with a third of the budget unspent. A best move
            # that has held for several iterations is unlikely to change, so it
            # gets no rope at all past the target.
            expected = (now - iteration_started) * 2.0
            iteration_started = now
            if elapsed + expected > soft * (1.3 if stable < 4 else 0.9):
                break
    except TimeUp:
        pass
    except Exception as exc:  # a crash here is an instant loss, so catch everything
        # Whatever went wrong, do not spend the rest of the game rediscovering
        # it. The Python evaluation is the one we can reason about, so drop to
        # it permanently and keep playing.
        global _score_position
        if _score_position is not _ordinary_score:
            _score_position = _ordinary_score
            print("dropped to the Python evaluation for the rest of the game")
        print(f"search error, falling back: {exc!r}")

    if best_move not in legal_moves:
        best_move = fallback
    _previous_spent_ms = (time.monotonic() - started) * 1000
    print(
        f"move {board.fullmove_number} d{depth} {best_move.uci()} {score:+d} "
        f"{_previous_spent_ms:.0f}ms of {soft * 1000:.0f}/{hard * 1000:.0f} "
        f"left {time_left_ms}ms inc {_increment_ms:.0f}ms"
    )
    return best_move.uci()

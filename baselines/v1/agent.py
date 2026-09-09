"""AI Chessathon submission.

Negamax with alpha-beta pruning, iterative deepening, a transposition table,
null-move pruning, capture/killer/history move ordering, and quiescence search,
on top of a material + piece-square evaluation that is tapered between
middlegame and endgame values as material comes off the board, plus enough
endgame knowledge to finish a won game: a lone king is driven to the edge, and
king-and-pawn against king is read off the solved table in `endgame`.

Import time runs once per game (see get_move's docstring for the per-call
contract); the module-level tables below are read-only after that point, and
the mutable state further down (transposition table, killer moves, history
heuristic, game history) is what is allowed to persist between our own moves
within a single game.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.polyglot

import endgame

# ---------------------------------------------------------------------------
# Evaluation
#
# A position's value is "material plus piece-square tables", computed twice
# (once with middlegame weights, once with endgame weights) and blended by
# how much material is left on the board. The idea: a knight belongs on a
# central outpost, a rook on an open file, a king behind its pawns while
# queens are on but marching up the board once they're gone. A single static
# table can't represent both, so we keep two and fade between them.
#
# Values are in centipawns (100 = one pawn) and follow the widely-taught
# "PeSTO" tapered scheme: same idea as material-only scoring, extended with
# a square-dependent bonus per piece per game phase.
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
# How much each piece counts towards "how far into the game are we". A full
# set of minor/major pieces sums to 24; an empty board sums to 0.
PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE = 24
# How much a legal move is worth to the side to move. This counts only the
# mover's own options rather than the difference between the two sides, which
# is cruder -- but the count is already to hand from generating the moves, and
# a per-piece difference measured with attack masks cost about a third of the
# search speed to compute, which is half a ply. Depth won that trade easily.
MOBILITY_WEIGHT = 2

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


MG_PST = {piece: _to_square_indexed(mg) for piece, (mg, _eg) in _RAW_PST.items()}
EG_PST = {piece: _to_square_indexed(eg) for piece, (_mg, eg) in _RAW_PST.items()}

MATE_SCORE = 100_000
DRAW_SCORE = 0


def _pawn_ending_score(board: chess.Board) -> float:
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


def _lone_king_score(board: chess.Board, loser: chess.Color) -> float:
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
    for _square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        material += sign * EG_VALUE[piece.piece_type]
    weak_king = chess.msb(board.kings & board.occupied_co[loser])
    strong_king = chess.msb(board.kings & board.occupied_co[not loser])
    drive = EDGE_WEIGHT * CENTRE_DISTANCE[weak_king] + APPROACH_WEIGHT * (
        14 - chess.square_manhattan_distance(strong_king, weak_king)
    )
    return material - drive if loser == chess.WHITE else material + drive


def evaluate(board: chess.Board, mobility: int) -> float:
    """Score `board` from the perspective of the side to move (negamax convention)."""
    if endgame.is_pawn_ending(board):
        return _pawn_ending_score(board)
    loser = _bare_king_side(board)
    if loser is not None:
        white_relative = _lone_king_score(board, loser)
        return white_relative if board.turn == chess.WHITE else -white_relative

    mg_score = 0
    eg_score = 0
    phase = 0
    for square, piece in board.piece_map().items():
        pt = piece.piece_type
        table_sq = square if piece.color == chess.WHITE else chess.square_mirror(square)
        sign = 1 if piece.color == chess.WHITE else -1
        mg_score += sign * (MG_VALUE[pt] + MG_PST[pt][table_sq])
        eg_score += sign * (EG_VALUE[pt] + EG_PST[pt][table_sq])
        phase += PHASE_WEIGHT[pt]
    phase = min(phase, MAX_PHASE)
    tapered = (mg_score * phase + eg_score * (MAX_PHASE - phase)) / MAX_PHASE
    # tapered is White-relative; flip to side-to-move-relative, then add mobility,
    # which is already counted from the mover's point of view.
    score = tapered if board.turn == chess.WHITE else -tapered
    return score + MOBILITY_WEIGHT * mobility


# ---------------------------------------------------------------------------
# Search state.
#
# These persist across our own moves within one game (a fresh process starts
# per game, per the contract), which is what makes a transposition table and
# a history heuristic worth having: work from an earlier move keeps paying
# off later in the same game.
# ---------------------------------------------------------------------------

EXACT, LOWER, UPPER = 0, 1, 2


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: int
    move: chess.Move | None


transposition_table: dict[int, TTEntry] = {}
killer_moves: dict[int, list[chess.Move]] = {}
history_heuristic: dict[tuple[bool, int, int], int] = {}
# Zobrist hash -> number of times this exact position (side to move, castling
# rights and en-passant possibilities included) has occurred in the real
# game so far. Search extends this while walking its own move tree, so a
# line that would complete a real threefold repetition scores as a draw
# instead of being invisible to a board built fresh from one FEN.
game_history: dict[int, int] = {}

MAX_DEPTH = 64
# No evaluation can reach this, so a score above it is a forced mate and nothing
# else. The margin is generous: it only has to clear the deepest ply we can
# reach, and quiescence goes deeper than MAX_DEPTH.
MATE_BOUND = MATE_SCORE - 1000


class TimeUp(Exception):
    pass


def _check_time(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeUp


def _score_to_tt(score: float, ply: int) -> float:
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


def _score_from_tt(score: float, ply: int) -> float:
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
#      attacker" -- taking a queen with a pawn is great, taking a pawn with
#      a queen is usually a trade you'd rather do differently)
#   3. "killer moves": quiet moves that caused a beta cutoff at this same
#      ply in a sibling branch, so they are worth trying again first
#   4. everything else, ranked by the history heuristic: quiet moves that
#      have caused cutoffs anywhere in the tree, tallied by (colour, from,
#      to) so the ranking survives across positions
# ---------------------------------------------------------------------------


def _order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None,
    ply: int,
) -> list[chess.Move]:
    killers = killer_moves.get(ply, ())

    def score(move: chess.Move) -> int:
        if move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            victim_piece = board.piece_type_at(move.to_square)
            victim = victim_piece if victim_piece is not None else chess.PAWN  # en passant
            attacker = board.piece_type_at(move.from_square)
            attacker_value = MG_VALUE[attacker] if attacker is not None else 0
            return 100_000 + MG_VALUE[victim] * 16 - attacker_value
        if move in killers:
            return 90_000 - killers.index(move)
        return history_heuristic.get((board.turn, move.from_square, move.to_square), 0)

    return sorted(moves, key=score, reverse=True)


def _record_cutoff(board: chess.Board, move: chess.Move, ply: int, depth: int) -> None:
    """A quiet move caused a beta cutoff: remember it for next time at this ply."""
    if board.is_capture(move):
        return
    killers = killer_moves.setdefault(ply, [])
    if move not in killers:
        killers.insert(0, move)
        del killers[2:]
    key = (board.turn, move.from_square, move.to_square)
    history_heuristic[key] = history_heuristic.get(key, 0) + depth * depth


def quiescence(board: chess.Board, alpha: float, beta: float, ply: int, deadline: float) -> float:
    """Extend the leaves with captures only, so we never score mid-exchange."""
    _check_time(deadline)
    moves = list(board.legal_moves)
    if not moves:
        # Counting the ply matters even down here: it is what makes a mate found
        # sooner score better than the same mate found later.
        return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE

    stand_pat = evaluate(board, len(moves))
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [m for m in moves if board.is_capture(m)]
    for move in _order_moves(board, captures, None, ply):
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1, deadline)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax(
    board: chess.Board,
    key: int,
    depth: int,
    alpha: float,
    beta: float,
    ply: int,
    deadline: float,
) -> float:
    _check_time(deadline)

    # A position seen twice already in the real game, reached a third time
    # anywhere in this search line, is a claimed draw -- score it as one so
    # the search does not walk straight past a repetition it can see coming.
    if game_history.get(key, 0) >= 3:
        return DRAW_SCORE
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return DRAW_SCORE

    alpha_orig = alpha
    entry = transposition_table.get(key)
    tt_move = entry.move if entry is not None else None
    if entry is not None and entry.depth >= depth:
        stored = _score_from_tt(entry.score, ply)
        if entry.flag == EXACT:
            return stored
        if entry.flag == LOWER and stored > alpha:
            alpha = stored
        elif entry.flag == UPPER and stored < beta:
            beta = stored
        if alpha >= beta:
            return stored

    moves = list(board.legal_moves)
    if not moves:
        return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE

    if depth <= 0:
        return quiescence(board, alpha, beta, ply, deadline)

    # Null-move pruning. Hand the opponent a free move: if our position is still
    # so strong that they cannot claw their way back to beta even with it, then
    # they will not manage it with a real move either, and the whole subtree can
    # go. Skipped in check, where passing is not a legal thing to imagine, and
    # skipped when we are down to king and pawns, where passing is genuinely bad
    # and the shortcut would prune away the defence that saves us. The null
    # position is deliberately kept out of `game_history`: it is not a position
    # the game ever stood in, and counting it would invent repetitions.
    if (
        depth >= 3
        and beta < MATE_BOUND
        and not board.is_check()
        and _has_non_pawn_material(board, board.turn)
    ):
        reduction = 2 + depth // 6
        board.push(chess.Move.null())
        try:
            null_key = chess.polyglot.zobrist_hash(board)
            score = -negamax(
                board, null_key, depth - 1 - reduction, -beta, -beta + 1, ply + 1, deadline
            )
        finally:
            board.pop()
        if score >= beta:
            return beta

    ordered = _order_moves(board, moves, tt_move, ply)
    best_score = -MATE_SCORE - 1.0
    best_move = ordered[0]
    for move in ordered:
        board.push(move)
        child_key = chess.polyglot.zobrist_hash(board)
        game_history[child_key] = game_history.get(child_key, 0) + 1
        try:
            score = -negamax(board, child_key, depth - 1, -beta, -alpha, ply + 1, deadline)
        finally:
            game_history[child_key] -= 1
            if game_history[child_key] == 0:
                del game_history[child_key]
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            _record_cutoff(board, move, ply, depth)
            break

    flag = EXACT
    if best_score <= alpha_orig:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    transposition_table[key] = TTEntry(depth, _score_to_tt(best_score, ply), flag, best_move)
    return best_score


def _root_search(
    board: chess.Board, key: int, depth: int, deadline: float
) -> tuple[float, chess.Move]:
    entry = transposition_table.get(key)
    tt_move = entry.move if entry is not None else None
    moves = _order_moves(board, list(board.legal_moves), tt_move, 0)

    alpha = -MATE_SCORE - 1.0
    beta = MATE_SCORE + 1.0
    best_score = alpha
    best_move = moves[0]
    for move in moves:
        board.push(move)
        child_key = chess.polyglot.zobrist_hash(board)
        game_history[child_key] = game_history.get(child_key, 0) + 1
        try:
            score = -negamax(board, child_key, depth - 1, -beta, -alpha, 1, deadline)
        finally:
            game_history[child_key] -= 1
            if game_history[child_key] == 0:
                del game_history[child_key]
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    transposition_table[key] = TTEntry(depth, best_score, EXACT, best_move)
    return best_score, best_move


# ---------------------------------------------------------------------------
# Time management.
#
# We get 120s base plus 0.5s per move, and time_left_ms is what's left before
# this move (the increment lands after). Spend roughly an even share of what
# remains, but never bet more than half of it on one move -- a long forcing
# line at move 20 shouldn't leave move 25 with nothing.
# ---------------------------------------------------------------------------

SAFETY_MARGIN_MS = 300
MIN_BUDGET_MS = 20


def _budget_seconds(time_left_ms: int, fullmove_number: int) -> float:
    usable_ms = max(0, time_left_ms - SAFETY_MARGIN_MS)
    moves_left_estimate = max(15, 45 - fullmove_number)
    budget_ms = usable_ms / moves_left_estimate
    budget_ms = min(budget_ms, usable_ms * 0.5)
    budget_ms = max(budget_ms, MIN_BUDGET_MS)
    return budget_ms / 1000


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation. See module docstring for the contract."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # The platform should never ask us to move with none available, but
        # returning something here is still better than raising.
        return chess.Move.null().uci()
    fallback = legal_moves[0]
    if len(legal_moves) == 1:
        return fallback.uci()

    root_key = chess.polyglot.zobrist_hash(board)
    game_history[root_key] = game_history.get(root_key, 0) + 1
    killer_moves.clear()
    if len(transposition_table) > 2_000_000:
        transposition_table.clear()

    best_move = fallback
    try:
        deadline = time.monotonic() + _budget_seconds(time_left_ms, board.fullmove_number)
        depth = 1
        while depth <= MAX_DEPTH:
            score, move = _root_search(board, root_key, depth, deadline)
            best_move = move
            if abs(score) >= MATE_BOUND:
                break  # forced mate found; deeper search can't improve on it
            depth += 1
            if time.monotonic() >= deadline:
                break
    except TimeUp:
        pass
    except Exception as exc:  # a crash here is an instant loss, so catch everything
        print(f"search error, falling back: {exc!r}")
        return best_move.uci()
    return best_move.uci()

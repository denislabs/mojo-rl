"""`RoomGenerator` — cellular-automata room generation (port of `roomgen.cpp`).

Shared by the room-based games (caveflyer, jumper). Operates on a `Grid` plus the
game's `out_of_bounds_object` (OOB grid reads return it). Faithful to the reference
draw-free structure: `update` (CA smoothing), `find_best_room` (largest connected
SPACE region via flood fill), `expand_room` (8-connected dilation), `find_path`
(4-connected BFS shortest path). Set membership is tracked with a `List[Bool]`
sized to the grid; scanning it ascending reproduces `std::set` iteration order.

No RNG here — the games draw the initial random fill + object placement; this
module only transforms the grid deterministically. See `docs/PROCGEN_PORT.md`.
"""

from .grid import Grid
from .object_ids import SPACE, WALL_OBJ


def _gget(grid: Grid, oob: Int, x: Int, y: Int) -> Int:
    if grid.contains(x, y):
        return grid.get(x, y)
    return oob


def _count_neighbors(grid: Grid, oob: Int, idx: Int, type: Int) -> Int:
    var x = idx % grid.w
    var y = idx // grid.w
    var n = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if _gget(grid, oob, x + i, y + j) == type:
                n += 1
    return n


def roomgen_update(mut grid: Grid, oob: Int):
    """One CA step: a cell becomes WALL_OBJ iff ≥5 of its 9-cell neighborhood
    (incl OOB) are walls, else SPACE."""
    var size = grid.w * grid.h
    var nxt = List[Int]()
    nxt.resize(size, 0)
    for i in range(size):
        if _count_neighbors(grid, oob, i, WALL_OBJ) >= 5:
            nxt[i] = WALL_OBJ
        else:
            nxt[i] = SPACE
    for i in range(size):
        grid.data[i] = nxt[i]


def _build_room(grid: Grid, idx: Int, mut member: List[Bool]) -> Int:
    """Flood-fill (4-connected) the connected SPACE region containing `idx`,
    marking `member[]`. Returns the cell count."""
    if grid.data[idx] != SPACE:
        return 0
    var queue = List[Int]()
    queue.append(idx)
    member[idx] = True
    var count = 0
    var head = 0
    while head < len(queue):
        var cur = queue[head]
        head += 1
        count += 1
        var x = cur % grid.w
        var y = cur // grid.w
        # 4-connected: (-1,0),(0,-1),(0,1),(1,0).
        var dxs: List[Int] = [-1, 0, 0, 1]
        var dys: List[Int] = [0, -1, 1, 0]
        for k in range(4):
            var nx = x + dxs[k]
            var ny = y + dys[k]
            if not grid.contains(nx, ny):
                continue
            var nidx = ny * grid.w + nx
            if not member[nidx] and grid.data[nidx] == SPACE:
                member[nidx] = True
                queue.append(nidx)
    return count


def roomgen_find_best_room(grid: Grid) -> List[Bool]:
    """Return a membership mask of the largest connected SPACE region (ties →
    lowest starting index, matching the reference's strict `>` scan)."""
    var size = grid.w * grid.h
    var covered = List[Bool]()
    covered.resize(size, False)
    var best = List[Bool]()
    best.resize(size, False)
    var best_size = -1
    for i in range(size):
        if grid.data[i] == SPACE and not covered[i]:
            var room = List[Bool]()
            room.resize(size, False)
            var cnt = _build_room(grid, i, room)
            for j in range(size):
                if room[j]:
                    covered[j] = True
            if cnt > best_size:
                best_size = cnt
                best = room^
    return best^


def roomgen_expand_room(grid: Grid, mut member: List[Bool], n: Int):
    """Dilate `member` by `n` rounds over 8-connected SPACE cells (grid state)."""
    var frontier = List[Int]()
    for i in range(len(member)):
        if member[i]:
            frontier.append(i)
    for _ in range(n):
        var nxt = List[Int]()
        for fi in range(len(frontier)):
            var cur = frontier[fi]
            if grid.data[cur] != SPACE:
                continue
            var x = cur % grid.w
            var y = cur // grid.w
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    var nx = x + i
                    var ny = y + j
                    if not grid.contains(nx, ny):
                        continue
                    var nidx = ny * grid.w + nx
                    if not member[nidx] and grid.data[nidx] == SPACE:
                        member[nidx] = True
                        nxt.append(nidx)
        frontier = nxt^


def roomgen_find_path(grid: Grid, src: Int, dst: Int) -> List[Int]:
    """4-connected BFS shortest path src→dst over SPACE cells (neighbor order
    left/down/up/right, matching the reference). Empty if unreachable."""
    var path = List[Int]()
    if grid.data[src] != SPACE:
        return path^
    var size = grid.w * grid.h
    var covered = List[Bool]()
    covered.resize(size, False)
    var expanded = List[Int]()
    var parents = List[Int]()
    expanded.append(src)
    parents.append(-1)
    var search_idx = 0
    while search_idx < len(expanded):
        var cur = expanded[search_idx]
        if cur == dst:
            break
        var x = cur % grid.w
        var y = cur // grid.w
        var dxs: List[Int] = [-1, 0, 0, 1]
        var dys: List[Int] = [0, -1, 1, 0]
        for k in range(4):
            var nx = x + dxs[k]
            var ny = y + dys[k]
            if not grid.contains(nx, ny):
                continue
            var nidx = ny * grid.w + nx
            if not covered[nidx] and grid.data[nidx] == SPACE:
                expanded.append(nidx)
                parents.append(search_idx)
                covered[nidx] = True
        search_idx += 1
    if search_idx < len(expanded) and expanded[search_idx] == dst:
        var tmp = List[Int]()
        var si = search_idx
        while si >= 0:
            tmp.append(expanded[si])
            si = parents[si]
        for j in range(len(tmp) - 1, -1, -1):
            path.append(tmp[j])
    return path^

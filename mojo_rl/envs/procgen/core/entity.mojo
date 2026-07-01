"""`Entity` — minimal value-type port of Procgen `entity.h`.

Phase-0 subset: only the fields the maze path touches (position, velocity,
radii, type, reflection). The full field set + `step()`/serialization arrive
with the Phase-1 `BasicAbstractGame` port.
"""


@fieldwise_init
struct Entity(Copyable, Movable):
    var x: Float32
    var y: Float32
    var vx: Float32
    var vy: Float32
    var rx: Float32
    var ry: Float32
    var type: Int
    var is_reflected: Bool

    @staticmethod
    def make(x: Float32, y: Float32, r: Float32, type: Int) -> Entity:
        return Entity(x, y, 0.0, 0.0, r, r, type, False)

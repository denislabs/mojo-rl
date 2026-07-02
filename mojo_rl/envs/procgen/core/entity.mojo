"""`Entity` — value-type port of Procgen `entity.{h,cpp}`.

The full field set + ctor defaults + `step()` used by the `BasicAbstractGame`
entity substrate (chaser and the other entity games). The maze path uses only a
subset (position/velocity/radii/type/reflection) via `Entity.make`, which is kept
as a convenience wrapper. EXPLOSION/TRAIL special-casing is present in reference
`step()` but unused so far — omitted until a game needs it.
"""

from .object_ids import EXPLOSION, EXPLOSION5


struct Entity(Copyable, Movable):
    var x: Float32
    var y: Float32
    var vx: Float32
    var vy: Float32
    var rx: Float32
    var ry: Float32
    var type: Int
    var image_type: Int
    var image_theme: Int
    var render_z: Int
    var will_erase: Bool
    var collides_with_entities: Bool
    var collision_margin: Float32
    var rotation: Float32
    var vrot: Float32
    var is_reflected: Bool
    var fire_time: Int
    var spawn_time: Int
    var life_time: Int
    var expire_time: Int
    var use_abs_coords: Bool
    var friction: Float32
    var smart_step: Bool
    var avoids_collisions: Bool
    var auto_erase: Bool
    var alpha: Float32
    var health: Float32
    var theta: Float32
    var grow_rate: Float32
    var alpha_decay: Float32

    def __init__(
        out self,
        x: Float32,
        y: Float32,
        vx: Float32,
        vy: Float32,
        rx: Float32,
        ry: Float32,
        type: Int,
    ):
        # Mirrors Entity::Entity(_x,_y,_vx,_vy,_rx,_ry,_type) defaults.
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.rx = rx
        self.ry = ry
        self.type = type
        self.image_type = type
        self.image_theme = 0
        self.render_z = 0
        self.will_erase = False
        self.collides_with_entities = False
        self.collision_margin = 0.0
        self.rotation = 0.0
        self.vrot = 0.0
        self.is_reflected = False
        self.fire_time = -1
        self.spawn_time = -1
        self.life_time = 0
        self.expire_time = -1
        self.use_abs_coords = False
        self.friction = 1.0
        self.smart_step = False
        self.avoids_collisions = False
        self.auto_erase = True
        self.alpha = 1.0
        self.health = 1.0
        self.theta = -100.0
        self.grow_rate = 1.0
        self.alpha_decay = 1.0

    @staticmethod
    def make(x: Float32, y: Float32, r: Float32, type: Int) -> Entity:
        return Entity(x, y, 0.0, 0.0, r, r, type)

    def step(mut self):
        # Entity::step(): smart_step entities are moved by basic_step_object, not
        # by their own velocity integration.
        if not self.smart_step:
            self.x += self.vx
            self.y += self.vy
        self.rotation += self.vrot
        self.vx *= self.friction
        self.vy *= self.friction
        self.life_time += 1
        if self.expire_time > 0 and self.life_time > self.expire_time:
            self.will_erase = True
        if self.type == EXPLOSION:
            if self.image_type < EXPLOSION5:
                self.image_type += 1
        self.rx *= self.grow_rate
        self.ry *= self.grow_rate
        self.alpha = self.alpha_decay * self.alpha

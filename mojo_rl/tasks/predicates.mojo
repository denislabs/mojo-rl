"""The goal language — `goal=In(brick, box_inside)` — P2.

A `.task`'s `goal=` is TEXT in `spec.mojo` and a small NUMERIC PROGRAM here.
Two steps, and the split is the design:

    parse_goal("In(brick, box_inside)")   -> a tree of terms holding NAMES
    bind_goal(goal, family, model)        -> the same tree holding INDICES

## ⚠⚠ THE NUMERIC PROGRAM IS THE POINT, AND IT IS FOR THE GPU

`TASK_LAYER_PLAN.md` §5.1: LIBERO calls `_check_success()` from Python every
step; at 1024 lanes a per-step host round-trip would dominate the step, so
**the goal predicate is part of the reward and runs where the reward runs.** A
`BoundGoal` is a fixed-capacity array of `(op, a, b, param)` — exactly the
shape that becomes P3's device tape, read by a switch in the reward kernel.
Nothing here allocates, nothing here holds a string.

That is also why binding is a separate pass: the strings are resolved ONCE, on
the host, at task-load time. A kernel never sees a name.

## ⚠ TIER A vs TIER B

* **Tier A** reads `Data.xpos` / `xquat` / `site_xpos` / `qpos` only — all
  per-lane and already on device (`Data.site_xpos` is `[BATCH, NSITE*3]`).
  In, On, Near, Above, Upright, Open, AtRegion.
* **Tier B** needs the contact array: Touching, Grasped.

**A task's `goal=` must be Tier A** — `require_tier_a` is that rule, and P3
calls it. Tier B parses and binds and evaluates on the CPU authoring path, and
may be logged as a diagnostic; it cannot train GPU-batched until per-lane
contact readback is specified, which is not in this plan.

Less restrictive than it sounds: LIBERO's `On(a,b)` reduces to
`b.check_ontop(a)`, and for a SITE target its `check_contact` returns True
unconditionally — its own comment says "There is no dynamics for site
objects". Its most-used predicate is already pure geometry against a site.

## ⚠ WHAT THIS FILE DOES NOT DO

It does not evaluate. Evaluation reads `Data` and belongs with the env wiring;
keeping the LANGUAGE separate is what lets the parser be gated without a
physics step, and what stops a second half-parser appearing in a kernel.
"""

from mojo_rl.core.kv import split_on
from .spec import FamilySpec


# ── ops ────────────────────────────────────────────────────────────────────
# ⚠ THE VALUES ARE THE WIRE FORMAT. A `BoundGoal` becomes a device tape in P3,
# so these numbers end up in GPU memory. Append; do not renumber.
comptime OP_IN: Int = 0
comptime OP_ON: Int = 1
comptime OP_NEAR: Int = 2
comptime OP_ABOVE: Int = 3
comptime OP_UPRIGHT: Int = 4
comptime OP_OPEN: Int = 5
comptime OP_AT_REGION: Int = 6
# Tier B — contacts.
comptime OP_TOUCHING: Int = 7
comptime OP_GRASPED: Int = 8
# Composition.
comptime OP_AND: Int = 9
comptime OP_OR: Int = 10
comptime OP_NOT: Int = 11

comptime OP_COUNT: Int = 12

# ⚠ A CAP, AND IT IS A DEVICE-SIDE ONE. P3's tape is `[N_TASKS, MAX_TERMS, 4]`
# and must be comptime-sized, so a goal that needs more terms than this cannot
# be expressed rather than silently truncated. Sixteen is far past anything
# LIBERO expresses (its goals are one or two predicates under an And).
comptime MAX_GOAL_TERMS: Int = 16


def op_from_name(s: String) raises -> Int:
    if s == "In":
        return OP_IN
    if s == "On":
        return OP_ON
    if s == "Near":
        return OP_NEAR
    if s == "Above":
        return OP_ABOVE
    if s == "Upright":
        return OP_UPRIGHT
    if s == "Open":
        return OP_OPEN
    if s == "AtRegion":
        return OP_AT_REGION
    if s == "Touching":
        return OP_TOUCHING
    if s == "Grasped":
        return OP_GRASPED
    if s == "And":
        return OP_AND
    if s == "Or":
        return OP_OR
    if s == "Not":
        return OP_NOT
    raise Error(
        "tasks: unknown predicate '" + s + "'. Known: In, On, Near, Above,"
        " Upright, Open, AtRegion, Touching, Grasped, And, Or, Not."
    )


def op_name(op: Int) -> String:
    if op == OP_IN:
        return String("In")
    if op == OP_ON:
        return String("On")
    if op == OP_NEAR:
        return String("Near")
    if op == OP_ABOVE:
        return String("Above")
    if op == OP_UPRIGHT:
        return String("Upright")
    if op == OP_OPEN:
        return String("Open")
    if op == OP_AT_REGION:
        return String("AtRegion")
    if op == OP_TOUCHING:
        return String("Touching")
    if op == OP_GRASPED:
        return String("Grasped")
    if op == OP_AND:
        return String("And")
    if op == OP_OR:
        return String("Or")
    return String("Not")


def op_arity(op: Int) -> Int:
    """How many arguments the predicate takes, INCLUDING its numeric one."""
    if op == OP_NEAR:
        return 3
    if op == OP_GRASPED or op == OP_NOT:
        return 1
    return 2


def op_is_tier_a(op: Int) -> Bool:
    """Can this run on device, reading only per-lane `Data`?"""
    return op != OP_TOUCHING and op != OP_GRASPED


def op_is_composite(op: Int) -> Bool:
    return op == OP_AND or op == OP_OR or op == OP_NOT


def op_takes_region(op: Int) -> Bool:
    """Is the SECOND argument a region name rather than a slot name?"""
    return op == OP_IN or op == OP_ON or op == OP_AT_REGION


def op_takes_number(op: Int) -> Bool:
    """Does the LAST argument parse as a number rather than a name?

    ⚠ `Near(a, b, d)` has THREE, the last numeric; `Upright(obj, tol)` and
    `Open(joint, frac)` have two, the last numeric. Getting this table wrong
    reads a distance as a body name and raises with a confusing message.
    """
    return op == OP_NEAR or op == OP_UPRIGHT or op == OP_OPEN


# ── the parsed, still-symbolic tree ────────────────────────────────────────


struct GoalTerm(Copyable, ImplicitlyCopyable, Movable):
    """One node. Leaves carry NAMES; composites carry child term indices.

    ⚠ POST-ORDER: a child's index is always LESS than its parent's, so the
    root is the last term and an evaluator can sweep the array forwards with
    no stack. That is what makes this shape usable inside a GPU kernel.
    """

    var op: Int
    var arg0: String
    var arg1: String
    var param: Float64
    var kid0: Int
    var kid1: Int

    def __init__(out self, op: Int):
        self.op = op
        self.arg0 = String("")
        self.arg1 = String("")
        self.param = 0.0
        self.kid0 = -1
        self.kid1 = -1


struct Goal(Movable & Deinitable):
    var terms: List[GoalTerm]

    def __init__(out self):
        self.terms = List[GoalTerm]()

    def __init__(out self, *, deinit move: Self):
        self.terms = move.terms^

    def root(self) -> Int:
        return len(self.terms) - 1

    def describe(self) raises -> String:
        return self._describe(self.root())

    def _describe(self, i: Int) raises -> String:
        ref t = self.terms[i]
        var s = op_name(t.op) + "("
        if op_is_composite(t.op):
            s += self._describe(t.kid0)
            if t.kid1 >= 0:
                s += ", " + self._describe(t.kid1)
        else:
            s += t.arg0
            if t.arg1.byte_length() > 0:
                s += ", " + t.arg1
            if op_takes_number(t.op):
                s += ", " + String(t.param)
        return s + ")"


struct _Parser(Movable & Deinitable):
    var src: String
    var pos: Int
    var terms: List[GoalTerm]

    def __init__(out self, src: String):
        self.src = src
        self.pos = 0
        self.terms = List[GoalTerm]()

    def __init__(out self, *, deinit move: Self):
        self.src = move.src^
        self.pos = move.pos
        self.terms = move.terms^

    def _at(self) -> String:
        if self.pos >= self.src.byte_length():
            return String("")
        return String(self.src[byte = self.pos : self.pos + 1])

    def _skip_ws(mut self):
        while self.pos < self.src.byte_length():
            var c = self._at()
            if c != " " and c != "\t" and c != "\n" and c != "\r":
                break
            self.pos += 1

    def _fail(self, why: String) raises:
        raise Error(
            "tasks: bad goal at offset " + String(self.pos) + ": " + why
            + " — in '" + self.src + "'"
        )

    def _ident(mut self) raises -> String:
        """A bare token: a name, a number, or a predicate head."""
        self._skip_ws()
        var start = self.pos
        while self.pos < self.src.byte_length():
            var c = self._at()
            if (
                c == "(" or c == ")" or c == "," or c == " " or c == "\t"
                or c == "\n" or c == "\r"
            ):
                break
            self.pos += 1
        if self.pos == start:
            self._fail(String("expected a name"))
        return String(self.src[byte=start : self.pos])

    def _expect(mut self, c: String) raises:
        self._skip_ws()
        if self._at() != c:
            self._fail(String("expected '") + c + "'")
        self.pos += 1

    def parse_expr(mut self) raises -> Int:
        """Parse one predicate; append its term; return the term's index."""
        if len(self.terms) >= MAX_GOAL_TERMS:
            raise Error(
                "tasks: goal has more than " + String(MAX_GOAL_TERMS)
                + " terms. That cap is a DEVICE-SIDE one — P3's tape is"
                " comptime-sized — so a longer goal cannot be expressed"
                " rather than silently truncated."
            )
        var head = self._ident()
        var op = op_from_name(head)
        self._expect(String("("))

        var t = GoalTerm(op)
        if op_is_composite(op):
            # ⚠ CHILDREN ARE PARSED FIRST, so their indices are lower than the
            # parent's and the array stays post-order. See GoalTerm.
            t.kid0 = self.parse_expr()
            if op != OP_NOT:
                self._expect(String(","))
                t.kid1 = self.parse_expr()
        else:
            var n = op_arity(op)
            t.arg0 = self._ident()
            if n >= 2:
                self._expect(String(","))
                if op_takes_number(op) and n == 2:
                    t.param = Float64(String(self._ident().strip()))
                else:
                    t.arg1 = self._ident()
            if n == 3:
                self._expect(String(","))
                t.param = Float64(String(self._ident().strip()))
        self._expect(String(")"))
        self.terms.append(t^)
        return len(self.terms) - 1


def parse_goal(text: String) raises -> Goal:
    """`In(brick, box_inside)` -> a post-order term tree.

    ⚠ TRAILING TEXT IS AN ERROR, not ignored. `In(a,b) On(c,d)` is two goals
    with no operator between them, and silently keeping the first would make a
    task succeed on half of what it says.
    """
    var p = _Parser(text)
    _ = p.parse_expr()
    p._skip_ws()
    # ⚠ RAISED INLINE, NOT VIA `p._fail`. Calling a method on `p` here and
    # then moving `p.terms` out below makes the compiler see a value that is
    # partially moved on one path and whole on the other — "field 'p.terms'
    # destroyed out of the middle of a value". Cheaper to spell the error.
    var leftover = p.pos != p.src.byte_length()
    var g = Goal()
    # ⚠ `.copy()`, NOT `^`. `_Parser` has a destructor, so moving a field out
    # leaves a partially-moved value the compiler refuses to destroy. The list
    # is capped at MAX_GOAL_TERMS (16) and this runs once at load time, so the
    # copy is free in every sense that matters.
    g.terms = p.terms.copy()
    if leftover:
        raise Error(
            "tasks: trailing text after the goal in '" + text + "'. Two"
            " predicates with no operator between them is not a goal —"
            " combine them with And(...) or Or(...)."
        )
    return g^


# ── the bound, numeric program — what a kernel would read ──────────────────


struct BoundTerm(Copyable, ImplicitlyCopyable, Movable):
    """`(op, a, b, param)`. No strings, no allocation, fixed width.

    `a` / `b` mean different things per op, and the table is the ABI:

        In / On / AtRegion   a = body id,   b = region index
        Near / Above         a = body id,   b = body id
        Upright              a = body id,   b = -1        param = tolerance
        Open                 a = joint id,  b = -1        param = fraction
        Touching             a = body id,   b = body id   (Tier B)
        Grasped              a = body id,   b = -1        (Tier B)
        And / Or             a = term idx,  b = term idx
        Not                  a = term idx,  b = -1
    """

    var op: Int
    var a: Int
    var b: Int
    var param: Float64

    def __init__(out self, op: Int, a: Int, b: Int, param: Float64):
        self.op = op
        self.a = a
        self.b = b
        self.param = param


struct BoundGoal(Movable & Deinitable):
    var terms: List[BoundTerm]

    def __init__(out self):
        self.terms = List[BoundTerm]()

    def __init__(out self, *, deinit move: Self):
        self.terms = move.terms^

    def root(self) -> Int:
        return len(self.terms) - 1

    def is_tier_a(self) -> Bool:
        for i in range(len(self.terms)):
            if not op_is_tier_a(self.terms[i].op):
                return False
        return True


def slot_body_id(slot: String, body_names: List[String]) raises -> Int:
    """The body a slot's instance contributes, by `Model` body id.

    ⚠⚠ THE PREFIX IS THE IDENTITY, AND THIS IS WHERE IT CASHES IN. §2.1: the
    `<attach prefix=>` string is the instance identity, so slot `brick` owns
    every body whose name begins `brick_` — in the composed SO-101 tabletop
    that is `brick_cube`, from the body named `cube` inside `cube.xml`.

    ⚠ THE LOWEST MATCHING ID IS THE SLOT'S ROOT. Bodies are emitted in tree
    order, so for a multi-body asset the first match is the root — the one a
    free joint attaches to and the one a goal means when it says "the brick".

    ⚠ `body_names[0]` IS THE WORLDBODY and these ids include it, matching
    `FlatModelDef.body_names`' own convention. Off-by-one here would name the
    wrong object and still resolve, which is the silent kind.
    """
    var want = slot + "_"
    for i in range(len(body_names)):
        if String(body_names[i]).startswith(want):
            return i
    raise Error(
        "tasks: slot '" + slot + "' has no body in the composed scene — no"
        " body name starts with '" + want + "'. Either the slot is not in the"
        " family's table, or the scene is stale (`pixi run"
        " gen-family-scenes`)."
    )


def site_id(name: String, site_names: List[String]) raises -> Int:
    for i in range(len(site_names)):
        if String(site_names[i]) == name:
            return i
    raise Error(
        "tasks: no site named '" + name + "' in the composed scene. ⚠ A"
        " region names the COMPOSED site — `<slot>_<site in the asset>` —"
        " because `<attach prefix=>` renames every element it splices."
    )


def bind_goal(
    g: Goal,
    f: FamilySpec,
    body_names: List[String],
    site_names: List[String],
) raises -> BoundGoal:
    """Resolve every name to an index. Runs ONCE, on the host, at load time.

    ⚠ REGIONS BIND TO THEIR INDEX IN THE FAMILY, not to a site id. The region
    carries its own site plus a rectangle, and the evaluator needs both; a
    goal term only has to say WHICH region. Resolving the site here would
    throw away the rectangle and quietly turn `In` into `AtSite`.
    """
    var out = BoundGoal()
    for i in range(len(g.terms)):
        ref t = g.terms[i]
        # ⚠ `-1` IS THE ABI's "no second argument", not a placeholder — see
        # BoundTerm's table. Every branch below sets `a`; `b` keeps this value
        # for the unary ops (Upright, Grasped, Not), and a kernel reads it as
        # "absent".
        var a: Int
        var b = -1
        if op_is_composite(t.op):
            a = t.kid0
            b = t.kid1
        elif t.op == OP_OPEN:
            # A joint, by the composed name. Left unresolved here: joints are
            # not slots, and the only user so far is a drawer/door family that
            # does not exist yet. Refused loudly rather than half-bound.
            raise Error(
                "tasks: Open(joint, frac) is not bound yet — no family in the"
                " tree has an articulated fixture. Add the joint lookup when"
                " one does, rather than guessing the convention now."
            )
        elif t.op == OP_AT_REGION:
            # ⚠ AT_REGION's FIRST ARGUMENT IS A SITE, NOT A SLOT. It is what
            # asks "is the gripper over the drop zone" — `robot_gripperframe`
            # is a site on the robot and belongs to no slot at all. Binding it
            # through `slot_body_id` like every other op would look for a body
            # named `robot_gripperframe_*` and raise on a goal that is
            # perfectly well formed.
            a = site_id(t.arg0, site_names)
            b = f.region_index(t.arg1)
            if b < 0:
                raise Error(
                    "tasks: AtRegion names region '" + t.arg1 + "', which"
                    " family '" + f.name + "' does not declare"
                )
        else:
            a = slot_body_id(t.arg0, body_names)
            if op_takes_region(t.op):
                b = f.region_index(t.arg1)
                if b < 0:
                    raise Error(
                        "tasks: goal names region '" + t.arg1 + "', which"
                        " family '" + f.name + "' does not declare"
                    )
            elif t.arg1.byte_length() > 0:
                b = slot_body_id(t.arg1, body_names)
        out.terms.append(BoundTerm(t.op, a, b, t.param))
    return out^


def require_tier_a(g: BoundGoal, task_name: String) raises:
    """⚠ `TASK_LAYER_PLAN.md` §5.1's RULE, made real. P3 calls this.

    A Tier B goal reads the contact array, which is not per-lane readable in
    the reward kernel today. Training against it GPU-batched would give a
    reward of whatever the default is — a flat curve, not a crash.
    """
    for i in range(len(g.terms)):
        if not op_is_tier_a(g.terms[i].op):
            raise Error(
                "task '" + task_name + "': goal uses "
                + op_name(g.terms[i].op) + ", which is TIER B (it reads"
                " contacts). Tier B runs on the CPU authoring path and may be"
                " logged as a diagnostic, but a task whose REWARD needs it"
                " cannot train GPU-batched — see TASK_LAYER_PLAN.md §5.1."
            )

"""READING LIBERO'S `.bddl` — P5's first half.

    var p = parse_bddl(read_text(path))
    p.problem   p.language   p.regions   p.fixtures   p.objects
    p.init      p.goal       p.interest

`TASK_LAYER_PLAN.md` §P5. LIBERO is **not a port target** and never was; it is
the cheapest available stress test of P1-P4 on somebody else's taxonomy. This
reads their format. `libero_import.mojo` translates what our own language can
express and REFUSES the rest by name.

## ⚠ THE GRAMMAR, MEASURED ON ALL 130 SHIPPED FILES

Not recalled — counted, with `references/LIBERO-master` in hand:

    (:domain 130)  (:language 130)  (:regions 130)  (:fixtures 130)
    (:objects 130) (:obj_of_interest 130) (:init 130) (:goal 130)
    (:target 1132) (:ranges 731) (:yaw_rotation 501)

    goal predicates : And 130, In 63, On 61, Close 11, Turnon 8, Open 7,
                      Turnoff 1
    init predicates : On 647, Open 22, Turnon 3, Close 3, In 1

⚠ AND THE TOKENISER IS SAFE BECAUSE THE CORPUS SAYS SO. Across all 130 files
every atom is drawn from `[A-Za-z0-9_.:-]` — no quotes, no escapes, no commas
— and `(:language ...)` is single-line and contains NO parenthesis in any of
them. That is what lets the free text be recovered by joining tokens to the
closing paren instead of needing a quoting rule. **Re-run the survey before
trusting this on a corpus that is not these 130 files.**

## ⚠⚠ A REGION NAME IS NOT UNIQUE — THE PAIR `(target, name)` IS

`open_the_middle_drawer_of_the_cabinet.bddl` declares `top_region` TWICE, once
targeting `wooden_cabinet_1` and once `wine_rack_1`. `:init` and `:goal` refer
to regions by the COMPOSED name `<target>_<name>` — `wooden_cabinet_1_top_
region` — which is what disambiguates them. Keying a map by the bare name
silently keeps whichever came last, and every goal naming the other one then
resolves to a rectangle somewhere else in the scene.

`composed_name()` is that pair, spelled once.

## ⚠ A REGION MAY HAVE NO `:ranges`, AND THAT IS NOT A DEFECT

731 `:ranges` against 1132 `:target`s. A region on the TABLE carries a
rectangle; a region on a FIXTURE — `wooden_cabinet_1_top_region`,
`flat_stove_1_cook_region` — carries none, because its geometry is a `<site>`
in the fixture's own asset XML (verified: `assets/articulated_objects/
wooden_cabinet.xml` declares `<site name="middle_region">`). `has_ranges` is
therefore a real distinction the importer acts on, not a parse failure.
"""

from mojo_rl.core.bytes import string_from_bytes


comptime BDDL_ATOM_CHARS: StaticString = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:"
)


def tokenize_bddl(text: String) raises -> List[String]:
    """`(`, `)` and atoms. Whitespace separates; nothing else is special.

    ⚠ BYTE-WISE, and the atoms are built as BYTES rather than by appending
    `chr(Int(b))` — the defect that corrupted fifteen readers in this tree
    (see `core/bytes.mojo`). Every `.bddl` atom measured is ASCII, but the
    `:language` free text is human prose and has no such guarantee on a corpus
    someone else writes.
    """
    var out = List[String]()
    var b = text.as_bytes()
    var cur = List[UInt8]()
    for i in range(len(b)):
        var c = b[i]
        var is_paren = c == UInt8(ord("(")) or c == UInt8(ord(")"))
        var is_space = (
            c == UInt8(ord(" ")) or c == UInt8(ord("\t"))
            or c == UInt8(ord("\n")) or c == UInt8(ord("\r"))
        )
        if is_paren or is_space:
            if len(cur) > 0:
                out.append(string_from_bytes(cur))
                cur = List[UInt8]()
            if is_paren:
                var p = List[UInt8]()
                p.append(c)
                out.append(string_from_bytes(p))
        else:
            cur.append(c)
    if len(cur) > 0:
        out.append(string_from_bytes(cur))
    return out^


struct BddlRegion(Movable & Deinitable):
    var name: String
    var target: String
    var has_ranges: Bool
    var x0: Float64
    var y0: Float64
    var x1: Float64
    var y1: Float64
    var has_yaw: Bool
    var yaw_lo: Float64
    var yaw_hi: Float64

    def __init__(out self, var name: String, var target: String):
        self.name = name^
        self.target = target^
        self.has_ranges = False
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.has_yaw = False
        self.yaw_lo = 0.0
        self.yaw_hi = 0.0

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.target = move.target^
        self.has_ranges = move.has_ranges
        self.x0 = move.x0
        self.y0 = move.y0
        self.x1 = move.x1
        self.y1 = move.y1
        self.has_yaw = move.has_yaw
        self.yaw_lo = move.yaw_lo
        self.yaw_hi = move.yaw_hi

    def composed_name(self) -> String:
        """`<target>_<name>` — how `:init` and `:goal` refer to this region.

        ⚠ THE ONLY SAFE KEY. A bare region name is NOT unique across a file;
        see the module header."""
        return String(self.target) + "_" + self.name


struct BddlNamed(Copyable, ImplicitlyCopyable, Movable):
    """One `instance - category` line of `:fixtures` or `:objects`."""

    var name: String
    var category: String

    def __init__(out self, var name: String, var category: String):
        self.name = name^
        self.category = category^


struct BddlAtom(Movable & Deinitable):
    """One predicate application: `(On plate_1 main_table_plate_region)`."""

    var pred: String
    var args: List[String]

    def __init__(out self, var pred: String, var args: List[String]):
        self.pred = pred^
        self.args = args^

    def __init__(out self, *, deinit move: Self):
        self.pred = move.pred^
        self.args = move.args^

    def show(self) -> String:
        var s = String("(") + self.pred
        for i in range(len(self.args)):
            s += " " + self.args[i]
        s += ")"
        return s^


struct BddlProblem(Movable & Deinitable):
    var problem: String
    var domain: String
    var language: String
    var regions: List[BddlRegion]
    var fixtures: List[BddlNamed]
    var objects: List[BddlNamed]
    var interest: List[String]
    var init: List[BddlAtom]
    var goal: List[BddlAtom]
    """The `:goal`'s terms, with the outer `(And ...)` UNWRAPPED.

    ⚠ EVERY ONE OF THE 130 FILES WRAPS ITS GOAL IN `And`, including the 68
    with a single term. Keeping the wrapper would make every imported goal a
    one-child conjunction and would spend a tape term on nothing —
    `MAX_TAPE_TERMS` is 3."""

    def __init__(out self):
        self.problem = String("")
        self.domain = String("")
        self.language = String("")
        self.regions = List[BddlRegion]()
        self.fixtures = List[BddlNamed]()
        self.objects = List[BddlNamed]()
        self.interest = List[String]()
        self.init = List[BddlAtom]()
        self.goal = List[BddlAtom]()

    def __init__(out self, *, deinit move: Self):
        self.problem = move.problem^
        self.domain = move.domain^
        self.language = move.language^
        self.regions = move.regions^
        self.fixtures = move.fixtures^
        self.objects = move.objects^
        self.interest = move.interest^
        self.init = move.init^
        self.goal = move.goal^

    def region_index(self, composed: String) raises -> Int:
        """Index of the region whose `<target>_<name>` is `composed`, or -1."""
        for i in range(len(self.regions)):
            if self.regions[i].composed_name() == composed:
                return i
        return -1

    def is_object(self, name: String) -> Bool:
        for i in range(len(self.objects)):
            if self.objects[i].name == name:
                return True
        return False

    def is_fixture(self, name: String) -> Bool:
        for i in range(len(self.fixtures)):
            if self.fixtures[i].name == name:
                return True
        return False


def _expect(t: List[String], i: Int, want: String, what: String) raises -> Int:
    if i >= len(t) or t[i] != want:
        raise Error(
            "bddl: expected '" + want + "' " + what + " at token "
            + String(i) + ", found '"
            + (String(t[i]) if i < len(t) else String("<end>")) + "'"
        )
    return i + 1


def _skip_form(t: List[String], start: Int) raises -> Int:
    """Index just past the balanced form beginning at `start` (a `(`)."""
    var depth = 0
    var i = start
    while i < len(t):
        if t[i] == "(":
            depth += 1
        elif t[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise Error("bddl: unbalanced parentheses from token " + String(start))


def _floats_in(t: List[String], start: Int) raises -> List[Float64]:
    """Every numeric atom inside the balanced form at `start`, EXCLUDING the
    form's own keyword.

    ⚠ `start` POINTS AT THE `(`, AND `start + 1` IS THE KEYWORD. Reading from
    `start` swept `:ranges` itself into the numbers and every one of the 130
    files failed to parse with "String is not convertible to float: ':ranges'".
    Loud, immediate, and caught by the survey's parse column — which is why
    that column is separate from the translate one.
    """
    var out = List[Float64]()
    var stop = _skip_form(t, start)
    for i in range(start + 2, stop):
        if t[i] != "(" and t[i] != ")":
            out.append(Float64(String(t[i])))
    return out^


def _atoms_in(t: List[String], start: Int, stop: Int) raises -> List[BddlAtom]:
    """Every sibling form in `[start, stop)`, as `(pred arg...)`.

    ⚠ ONE LEVEL DEEP, WHICH IS ALL THE CORPUS HAS: a predicate's arguments are
    always bare atoms, never nested forms. A nested one would land in `args`
    as its bare tokens and read as extra arguments, so `translate` checks the
    ARITY it expects rather than trusting the shape.
    """
    var out = List[BddlAtom]()
    var j = start
    while j < stop and t[j] == "(":
        var astop = _skip_form(t, j)
        var pred = String(t[j + 1])
        var args = List[String]()
        for a in range(j + 2, astop - 1):
            args.append(String(t[a]))
        out.append(BddlAtom(pred^, args^))
        j = astop
    return out^


def parse_bddl(text: String) raises -> BddlProblem:
    """One `.bddl` file. RAISES on anything it does not recognise.

    ⚠⚠ UNKNOWN BLOCKS RAISE RATHER THAN BEING SKIPPED. `tasks/spec.mojo` made
    the same choice for the same reason: a silently dropped `:goal` is a task
    that always succeeds, and a silently dropped `:regions` entry is a goal
    that resolves to somebody else's rectangle. The corpus has exactly eight
    block keywords; a ninth means this reader has not seen the file it is
    reading.
    """
    var t = tokenize_bddl(text)
    var p = BddlProblem()
    var i = 0
    i = _expect(t, i, String("("), "to open the file")
    i = _expect(t, i, String("define"), "as the first form")
    i = _expect(t, i, String("("), "to open (problem ...)")
    i = _expect(t, i, String("problem"), "as the problem header")
    if i >= len(t):
        raise Error("bddl: file ends inside (problem ...)")
    p.problem = String(t[i])
    i += 1
    i = _expect(t, i, String(")"), "to close (problem ...)")

    while i < len(t) and t[i] == "(":
        var key = String(t[i + 1]) if i + 1 < len(t) else String("")
        var stop = _skip_form(t, i)
        var j = i + 2
        if key == ":domain":
            p.domain = String(t[j])
        elif key == ":language":
            # ⚠ FREE TEXT, JOINED BACK TOGETHER. Safe because no `:language`
            # in the corpus contains a parenthesis — see the module header.
            var s = String("")
            while j < stop - 1:
                if s.byte_length() > 0:
                    s += " "
                s += String(t[j])
                j += 1
            p.language = s^
        elif key == ":regions":
            while j < stop - 1 and t[j] == "(":
                var rstop = _skip_form(t, j)
                var nm = String(t[j + 1])
                var k = j + 2
                var target = String("")
                var r_lo = -1
                var y_lo = -1
                while k < rstop - 1 and t[k] == "(":
                    var sub = String(t[k + 1])
                    var sstop = _skip_form(t, k)
                    if sub == ":target":
                        target = String(t[k + 2])
                    elif sub == ":ranges":
                        r_lo = k
                    elif sub == ":yaw_rotation":
                        y_lo = k
                    else:
                        raise Error(
                            "bddl: region '" + nm + "' has an unknown"
                            " sub-block '" + sub + "'. The corpus has"
                            " :target, :ranges and :yaw_rotation."
                        )
                    k = sstop
                if target == "":
                    raise Error(
                        "bddl: region '" + nm + "' has no (:target ...). Its"
                        " composed name is <target>_<name> and every :init"
                        " and :goal reference uses it, so a region without a"
                        " target can never be named."
                    )
                var reg = BddlRegion(nm^, target^)
                if r_lo >= 0:
                    var f = _floats_in(t, r_lo)
                    if len(f) != 4:
                        raise Error(
                            "bddl: region '" + reg.name + "' has "
                            + String(len(f)) + " range values; a rectangle is"
                            " four (x0 y0 x1 y1). Multiple alternative ranges"
                            " are not supported — a sampler choosing between"
                            " them is a different distribution."
                        )
                    reg.has_ranges = True
                    reg.x0 = f[0]
                    reg.y0 = f[1]
                    reg.x1 = f[2]
                    reg.y1 = f[3]
                if y_lo >= 0:
                    var g = _floats_in(t, y_lo)
                    if len(g) == 2:
                        reg.has_yaw = True
                        reg.yaw_lo = g[0]
                        reg.yaw_hi = g[1]
                p.regions.append(reg^)
                j = rstop
        elif key == ":fixtures" or key == ":objects":
            # ⚠⚠ A PDDL **TYPED LIST**: `name1 name2 ... - category`, and the
            # names before the dash may be SEVERAL. 40 of the 130 files use
            # the plural form — `moka_pot_1 moka_pot_2 - moka_pot` — and a
            # reader that assumed one name per dash failed all 40 loudly.
            # (Loudly is the point: it was a parse error naming the entry, not
            # a silently dropped second instance, which would have been a
            # scene missing an object nothing complained about.)
            var pending = List[String]()
            while j < stop - 1:
                if t[j] == "-":
                    if j + 1 >= stop - 1:
                        raise Error(
                            "bddl: '" + key + "' ends after '-' with no"
                            " category."
                        )
                    if len(pending) == 0:
                        raise Error(
                            "bddl: '" + key + "' has a '-' with no instance"
                            " names before it."
                        )
                    var cat = String(t[j + 1])
                    for q in range(len(pending)):
                        if key == ":fixtures":
                            p.fixtures.append(
                                BddlNamed(String(pending[q]), String(cat))
                            )
                        else:
                            p.objects.append(
                                BddlNamed(String(pending[q]), String(cat))
                            )
                    pending = List[String]()
                    j += 2
                else:
                    pending.append(String(t[j]))
                    j += 1
            if len(pending) != 0:
                raise Error(
                    "bddl: '" + key + "' ends with " + String(len(pending))
                    + " instance name(s) that no '- <category>' claims. A"
                    " dropped instance is an object missing from the scene."
                )
        elif key == ":obj_of_interest":
            while j < stop - 1:
                p.interest.append(String(t[j]))
                j += 1
        elif key == ":init":
            p.init = _atoms_in(t, j, stop - 1)
        elif key == ":goal":
            var top = _atoms_in(t, j, stop - 1)
            if len(top) == 1 and top[0].pred == "And":
                # ⚠ UNWRAP THE OUTER `And` — all 130 files have one, including
                # the 68 whose goal is a single term. Keeping it would spend a
                # tape term on nothing, and MAX_TAPE_TERMS is 3.
                var and_start = j
                while and_start < stop - 1 and t[and_start] != "(":
                    and_start += 1
                var and_stop = _skip_form(t, and_start)
                p.goal = _atoms_in(t, and_start + 2, and_stop - 1)
                if len(p.goal) == 0:
                    raise Error(
                        "bddl: problem '" + p.problem + "' has (And) with no"
                        " terms. An empty conjunction is TRUE, i.e. a task"
                        " that succeeds at reset."
                    )
            else:
                p.goal = top^
        else:
            raise Error(
                "bddl: unknown block '" + key + "'. The 130-file corpus has"
                " :domain, :language, :regions, :fixtures, :objects,"
                " :obj_of_interest, :init and :goal. Refused rather than"
                " skipped — a dropped block is a task that means something"
                " else."
            )
        i = stop
    if len(p.goal) == 0:
        raise Error(
            "bddl: problem '" + p.problem + "' has an EMPTY goal. A task with"
            " no goal is one that never succeeds, which reads as a policy"
            " that cannot learn."
        )
    return p^

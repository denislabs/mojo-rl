"""Which rows of a `.demo` file a learner takes — `--demo-filter`."""

from .file import DemoSet


struct DemoFilter(Copyable, Movable, ImplicitlyCopyable):
    """`all` (default), `success` (rows of successful episodes only) or
    `intervened` (only the steps where a human or an expert overrode a
    policy — the DAgger / intervention half of HIL-SERL)."""

    comptime ALL = 0
    comptime SUCCESS = 1
    comptime INTERVENED = 2

    var kind: Int

    def __init__(out self, kind: Int = Self.ALL):
        self.kind = kind

    @staticmethod
    def parse(text: String, who: String = "demos") raises -> DemoFilter:
        if text == "all":
            return DemoFilter(Self.ALL)
        if text == "success":
            return DemoFilter(Self.SUCCESS)
        if text == "intervened":
            return DemoFilter(Self.INTERVENED)
        raise Error(
            who + ": --demo-filter must be all, success or intervened (got '"
            + text + "')"
        )

    def name(self) -> String:
        if self.kind == Self.SUCCESS:
            return "success"
        if self.kind == Self.INTERVENED:
            return "intervened"
        return "all"

    def keeps(self, ref ds: DemoSet, r: Int) -> Bool:
        if self.kind == Self.SUCCESS:
            return ds.row_success(r)
        if self.kind == Self.INTERVENED:
            return ds.row_intervened(r)
        return True

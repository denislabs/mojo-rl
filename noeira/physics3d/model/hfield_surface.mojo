"""Where a `<hfield>` sample sits in the geom's local frame.

THREE LINES OF ARITHMETIC WITH TWO CONSUMERS, which is exactly why they live
here rather than inline in both:

  · `ray/hfield.mojo` (`mj_rayHfield`) builds the triangles a ray is tested
    against — the surface the PHYSICS and the rangefinders see;
  · `render/renderer3d.mojo` (`draw_heightfield`) builds the triangles the
    viewer draws — the surface a HUMAN sees.

⚠⚠ THE TWO DISAGREEING IS THE WORST OUTCOME AVAILABLE, and it is silent. The
viewer exists to answer "is the model built and posed the way I think"; a
drawn surface that is a half-cell off, or transposed in row/column, looks
entirely plausible and quietly contradicts the terrain the robot is standing
on. Nothing else in the picture would look wrong.
[[feedback_a_rule_written_inline_twice_drifts]]

⚠ `ray/hfield.mojo` STILL SPELLS THE ARITHMETIC INLINE, and deliberately: it is
generic over the model's `DTYPE` and runs inside a GPU kernel, where these
values are already in registers as part of the cell walk. What binds the two is
`tests/physics3d/test_hfield_surface_matches_ray.mojo`, which drops a vertical
ray onto nodes computed HERE and asserts the ray comes back at this height. A
divergence fails that gate rather than becoming a rendering curiosity.

THE CONVENTION, which is MuJoCo's:

    x = 2*size_x*c/(ncol-1) - size_x        c indexes COLUMNS, the x axis
    y = 2*size_y*r/(nrow-1) - size_y        r indexes ROWS,    the y axis
    z = data[adr + r*ncol + c] * size_z     data is NORMALISED to [0, 1]

⚠ `size_z` SCALES A UNIT GRID. A field holding metres would be wrong by the
elevation scale everywhere and right at exactly one value of it.

⚠ THE BASE IS NOT PART OF THIS. A `<hfield>` is this surface sitting on a solid
box that extends `size[3]` BELOW z = 0; nothing here knows about it, and both
consumers add it themselves.
"""


@always_inline
def hfield_node_x(c: Int, ncol: Int, size_x: Float64) -> Float64:
    """Local x of grid column `c`. `ncol` must be at least 2."""
    return (2.0 * size_x) * Float64(c) / Float64(ncol - 1) - size_x


@always_inline
def hfield_node_y(r: Int, nrow: Int, size_y: Float64) -> Float64:
    """Local y of grid row `r`. `nrow` must be at least 2."""
    return (2.0 * size_y) * Float64(r) / Float64(nrow - 1) - size_y


@always_inline
def hfield_node_z(
    grid: List[Float64],
    adr: Int,
    ncol: Int,
    r: Int,
    c: Int,
    size_z: Float64,
) -> Float64:
    """Local z of grid node (r, c) — the elevation, in metres.

    ⚠ ROW-MAJOR WITH `ncol` AS THE STRIDE. Transposing this is the mistake the
    module docstring warns about: on a SQUARE grid it still indexes in range,
    still produces a bowl, and is wrong everywhere the terrain is not
    symmetric.
    """
    return grid[adr + r * ncol + c] * size_z

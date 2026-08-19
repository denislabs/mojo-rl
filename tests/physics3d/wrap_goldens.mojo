"""MuJoCo 3.10.0 goldens for `mju_wrap` — GENERATED, do not edit.

Regenerate: pixi run python scripts/dump_mujoco_wrap.py

`mju_wrap` is not exposed to the Python bindings, so each row is read
off a real one-tendon model: the wrap POINTS come straight from
`d.wrap_xpos`, and the arc length is `ten_length` minus the two
straight runs. See scripts/dump_mujoco_wrap.py for the poses and why
each one is there.

Columns (31 per case):
   0- 2  x0            6- 8  geom pos      18     radius
   3- 5  x1            9-17  geom mat      19     wtype 1=sph 2=cyl
  20     has_side     21-23  sidesite      24     wlen (-1 = no wrap)
  25-27  wrap point 0 28-30  wrap point 1
"""

comptime WRAP_COLS: Int = 31


def wrap_case_labels() -> List[String]:
    var v = List[String]()
    v.append(String("sphere clear (no wrap) [wrapnum 2]"))
    v.append(String("sphere blocked [wrapnum 4]"))
    v.append(String("sphere blocked, asymmetric [wrapnum 4]"))
    v.append(String("sphere collinear through centre [wrapnum 4]"))
    v.append(String("sphere offset centre [wrapnum 4]"))
    v.append(String("sphere sidesite +y [wrapnum 4]"))
    v.append(String("sphere sidesite -y (the long way) [wrapnum 4]"))
    v.append(String("sphere sidesite, clear segment [wrapnum 4]"))
    v.append(String("sphere sidesite inside [wrapnum 4]"))
    v.append(String("sphere sidesite inside, asymmetric [wrapnum 4]"))
    v.append(String("cylinder clear [wrapnum 2]"))
    v.append(String("cylinder blocked, flat [wrapnum 4]"))
    v.append(String("cylinder blocked, helix [wrapnum 4]"))
    v.append(String("cylinder rotated 90 about x [wrapnum 4]"))
    v.append(String("cylinder sidesite +y [wrapnum 4]"))
    v.append(String("cylinder sidesite -y (long way) [wrapnum 4]"))
    v.append(String("cylinder sidesite inside [wrapnum 4]"))
    v.append(String("cylinder offset + rotated [wrapnum 4]"))
    return v^


def wrap_goldens() -> List[Float64]:
    var v = List[Float64]()
    # sphere clear (no wrap)
    v.append(-0.5); v.append(0.4); v.append(0.0); v.append(0.5);
    v.append(0.4); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(-1.0); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(0.0); v.append(0.0);
    # sphere blocked
    v.append(-0.5); v.append(0.02); v.append(0.0); v.append(0.5);
    v.append(0.02); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(0.03224322655268891); v.append(-0.016051868794846245); v.append(0.09870328012884394); v.append(0.0);
    v.append(0.01605186879484624); v.append(0.09870328012884395); v.append(0.0);
    # sphere blocked, asymmetric
    v.append(-0.4); v.append(0.05); v.append(0.03); v.append(0.6);
    v.append(-0.02); v.append(-0.05); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(0.0325573609263784); v.append(-0.01323676786581698); v.append(0.09880956443505681); v.append(-0.007839512269320994);
    v.append(0.01904367508736536); v.append(0.0976048831956543); v.append(-0.010517852229877424);
    # sphere collinear through centre
    v.append(-0.5); v.append(0.0); v.append(0.0); v.append(0.5);
    v.append(0.0); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(0.04027158415806609); v.append(-0.020000000000000004); v.append(0.0692820323027551); v.append(0.0692820323027551);
    v.append(0.020000000000000004); v.append(0.0692820323027551); v.append(0.0692820323027551);
    # sphere offset centre
    v.append(-0.5); v.append(0.05); v.append(0.0); v.append(0.5);
    v.append(0.05); v.append(0.0); v.append(0.1); v.append(0.05);
    v.append(-0.02); v.append(0.7071067829175877); v.append(-0.7071067794555074); v.append(0.0);
    v.append(0.7071067794555074); v.append(0.7071067829175877); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.08); v.append(1.0);
    v.append(0.0); v.append(-0.04142135589110146); v.append(0.19142135658351755); v.append(-0.02);
    v.append(0.02011846149097063); v.append(0.0919865881963407); v.append(0.050000000000000086); v.append(0.05959764589022071);
    v.append(0.11204560261938183); v.append(0.050000000000000086); v.append(0.05908794761236344);
    # sphere sidesite +y
    v.append(-0.5); v.append(0.02); v.append(0.0); v.append(0.5);
    v.append(0.02); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(1.0); v.append(0.0); v.append(0.3); v.append(0.0);
    v.append(0.03224322655268891); v.append(-0.016051868794846245); v.append(0.09870328012884394); v.append(0.0);
    v.append(0.01605186879484624); v.append(0.09870328012884395); v.append(0.0);
    # sphere sidesite -y (the long way)
    v.append(-0.5); v.append(0.02); v.append(0.0); v.append(0.5);
    v.append(0.02); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(1.0); v.append(0.0); v.append(-0.3); v.append(0.0);
    v.append(0.04823470140200453); v.append(-0.023884233441575486); v.append(-0.09710583603938708); v.append(0.0);
    v.append(0.02388423344157549); v.append(-0.09710583603938709); v.append(0.0);
    # sphere sidesite, clear segment
    v.append(-0.5); v.append(0.4); v.append(0.0); v.append(0.5);
    v.append(0.4); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(1.0);
    v.append(1.0); v.append(0.0); v.append(-0.3); v.append(0.0);
    v.append(0.16631132551359085); v.append(-0.07389810068621229); v.append(-0.06737262585776535); v.append(0.0);
    v.append(0.0738981006862123); v.append(-0.06737262585776535); v.append(0.0);
    # sphere sidesite inside
    v.append(-0.5); v.append(0.3); v.append(0.0); v.append(0.5);
    v.append(0.3); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.2); v.append(1.0);
    v.append(1.0); v.append(0.0); v.append(0.05); v.append(0.0);
    v.append(0.0); v.append(5.820339182838197e-09); v.append(0.19999999999999996); v.append(0.0);
    v.append(5.820339182838197e-09); v.append(0.19999999999999996); v.append(0.0);
    # sphere sidesite inside, asymmetric
    v.append(-0.6); v.append(0.35); v.append(0.05); v.append(0.45);
    v.append(0.28); v.append(-0.04); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.2); v.append(1.0);
    v.append(1.0); v.append(0.02); v.append(0.05); v.append(0.01);
    v.append(0.0); v.append(0.006287872580535286); v.append(0.199895785365284); v.append(-0.0014620709993945127);
    v.append(0.006287872580535286); v.append(0.199895785365284); v.append(-0.0014620709993945127);
    # cylinder clear
    v.append(-0.5); v.append(0.4); v.append(0.0); v.append(0.5);
    v.append(0.4); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(2.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(-1.0); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(0.0); v.append(0.0);
    # cylinder blocked, flat
    v.append(-0.5); v.append(0.02); v.append(0.0); v.append(0.5);
    v.append(0.02); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(2.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(0.0322432265526888); v.append(-0.016051868794846245); v.append(0.09870328012884394); v.append(0.0);
    v.append(0.016051868794846245); v.append(0.09870328012884394); v.append(0.0);
    # cylinder blocked, helix
    v.append(-0.5); v.append(0.02); v.append(-0.2); v.append(0.5);
    v.append(0.02); v.append(0.25); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(2.0);
    v.append(0.0); v.append(0.0); v.append(0.2); v.append(0.0);
    v.append(0.03528228693225377); v.append(-0.016051868794846245); v.append(0.09870328012884394); v.append(0.0178373518751464);
    v.append(0.016051868794846245); v.append(0.09870328012884394); v.append(0.03216264812485359);
    # cylinder rotated 90 about x
    v.append(-0.5); v.append(0.0); v.append(0.02); v.append(0.5);
    v.append(0.0); v.append(0.02); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0000000000000002); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(0.0); v.append(-1.0000000000000002); v.append(0.0);
    v.append(1.0000000000000002); v.append(0.0); v.append(0.1); v.append(2.0);
    v.append(0.0); v.append(0.0); v.append(-0.20000000000000007); v.append(0.0);
    v.append(0.032243226552688686); v.append(-0.01605186879484624); v.append(0.0); v.append(0.098703280128844);
    v.append(0.01605186879484624); v.append(0.0); v.append(0.098703280128844);
    # cylinder sidesite +y
    v.append(-0.5); v.append(0.02); v.append(0.05); v.append(0.5);
    v.append(0.02); v.append(-0.05); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(2.0);
    v.append(1.0); v.append(0.0); v.append(0.3); v.append(0.0);
    v.append(0.03239999519725173); v.append(-0.016051868794846245); v.append(0.09870328012884394); v.append(0.0015916995833007985);
    v.append(0.016051868794846245); v.append(0.09870328012884394); v.append(-0.0015916995833007985);
    # cylinder sidesite -y (long way)
    v.append(-0.5); v.append(0.02); v.append(0.05); v.append(0.5);
    v.append(0.02); v.append(-0.05); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.1); v.append(2.0);
    v.append(1.0); v.append(0.0); v.append(-0.3); v.append(0.0);
    v.append(0.04846200490197283); v.append(-0.023884233441575486); v.append(-0.09710583603938708); v.append(0.0023441149597598354);
    v.append(0.023884233441575486); v.append(-0.09710583603938708); v.append(-0.0023441149597598424);
    # cylinder sidesite inside
    v.append(-0.6); v.append(0.4); v.append(0.0); v.append(0.6);
    v.append(0.4); v.append(0.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.25); v.append(2.0);
    v.append(1.0); v.append(0.0); v.append(0.05); v.append(0.0);
    v.append(0.0); v.append(-1.7727027584983637e-08); v.append(0.24999999999999933); v.append(0.0);
    v.append(-1.7727027584983637e-08); v.append(0.24999999999999933); v.append(0.0);
    # cylinder offset + rotated
    v.append(-0.5); v.append(-0.02); v.append(0.04); v.append(0.5);
    v.append(0.01); v.append(-0.03); v.append(0.05); v.append(-0.03);
    v.append(0.01); v.append(0.7071067829175877); v.append(-0.7071067794555074); v.append(0.0);
    v.append(0.7071067794555074); v.append(0.7071067829175877); v.append(0.0); v.append(0.0);
    v.append(0.0); v.append(1.0); v.append(0.07); v.append(2.0);
    v.append(0.0); v.append(-0.09142135589110147); v.append(0.11142135658351754); v.append(0.01);
    v.append(0.012373283212791297); v.append(0.0423560252486441); v.append(0.039581388675425516); v.append(0.0019694665237938638);
    v.append(0.05468059205366195); v.append(0.039843339396303204); v.append(0.0011091157814229601);
    return v^


comptime ARM_COLS: Int = 5   # angle, len_cyl, wrapnum_cyl, len_sph, wrapnum_sph
comptime SOFTFOOT_COLS: Int = 6  # uniform qpos offset, then 5 tendon lengths


def arm_sweep_goldens() -> List[Float64]:
    """`tests/physics3d/assets/wrap_arm.xml`, hinge swept -1.5 .. +1.8 rad."""
    var v = List[Float64]()
    v.append(-1.5); v.append(0.735000527088424); v.append(2.0); v.append(0.8871597183202067); v.append(4.0);
    v.append(-1.2); v.append(0.7833991027108225); v.append(2.0); v.append(0.9049463337570236); v.append(4.0);
    v.append(-0.9); v.append(0.8242407549679379); v.append(4.0); v.append(0.915893597023272); v.append(4.0);
    v.append(-0.6000000000000001); v.append(0.8561541358757438); v.append(4.0); v.append(0.9195200101506488); v.append(4.0);
    v.append(-0.30000000000000004); v.append(0.8780213259047938); v.append(4.0); v.append(0.9155346868520373); v.append(4.0);
    v.append(0.0); v.append(0.8891264692265233); v.append(4.0); v.append(0.9038341592442873); v.append(4.0);
    v.append(0.2999999999999998); v.append(0.8780213259047938); v.append(4.0); v.append(0.8845003516711939); v.append(4.0);
    v.append(0.6000000000000001); v.append(0.8561541358757438); v.append(4.0); v.append(0.8577976780747893); v.append(4.0);
    v.append(0.8999999999999999); v.append(0.8242407549679379); v.append(4.0); v.append(0.8241691522665529); v.append(4.0);
    v.append(1.1999999999999997); v.append(0.7833991027108225); v.append(2.0); v.append(0.7833991027108225); v.append(2.0);
    v.append(1.5); v.append(0.735000527088424); v.append(2.0); v.append(0.735000527088424); v.append(2.0);
    v.append(1.7999999999999998); v.append(0.6820154486060723); v.append(2.0); v.append(0.6820154486060723); v.append(2.0);
    return v^


def softfoot_goldens() -> List[Float64]:
    """iit_softfoot at qpos0 + a uniform offset, 5 tendons each."""
    var v = List[Float64]()
    v.append(0.0); v.append(0.2840427979125473); v.append(0.2840427979125473); v.append(0.2840427979125473); v.append(0.28405210317431384); v.append(0.2840427979125473);
    v.append(0.2); v.append(0.27026692170483346); v.append(0.27026692170483346); v.append(0.27026692170483346); v.append(0.2702775497629539); v.append(0.27026692170483346);
    v.append(0.5); v.append(0.2506803591055375); v.append(0.2506803591055375); v.append(0.2506803591055375); v.append(0.2506937013247338); v.append(0.2506803591055375);
    return v^

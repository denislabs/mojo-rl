"""`dm_control` `humanoid_CMU` model — port of `dm_control/suite/humanoid_CMU.xml`.

Verbatim apart from the `<include>` lines and the `<sensor>` block. The body of
this file was EXTRACTED FROM THE REFERENCE PROGRAMMATICALLY rather than
transcribed, so the 17 kB of joint ranges and unnormalized quaternions below
cannot carry a typo the eye would miss.

THE SENSOR BLOCK IS DROPPED, exactly as in `humanoid_xml`. The XML declares
eight sensors and the three tasks read ONE of them — `grep sensordata
humanoid_CMU.py` returns a single line, `thorax_subtreelinvel`, which we
compute from `Data.xvel` via `sensors.subtree_linvel`. (`merge_mjcf` does not
carry a `<sensor>` section anyway.) All five SITES stay, so `nsite` still
matches MuJoCo's 5 and a future `touch` port has its zones.

WHAT THIS MODEL EXERCISES THAT NO EARLIER PORTED DOMAIN DOES:

  * A NAMED TOP-LEVEL DEFAULT CLASS, `<default class="main">`. It is the only
    one in all nineteen suite domains — every other model opens with a bare
    `<default>`. `_extract_section_inner` is depth-counted so the block is
    read whole, and `_strip_nested_defaults` removes the nested `humanoid`
    class from the root lookup; but this is the first model that proves it,
    and a regression here would silently hand every geom and joint the WRONG
    DEFAULTS rather than fail.
  * FIFTY-SIX ACTUATORS AND FIFTY-SEVEN JOINTS, against a comptime parser that
    recorded 32 of each until 2026-08-03. Both scans were `while count < CAP`
    while `ParsedModel` counted the tags independently, so before the widening
    this model would have built, exposed all 56 controls, and silently applied
    zero force through 24 of them. See `MAX_COMPTIME_ACTUATORS`.

WARNING: COUNT MODEL ELEMENTS WITH MuJoCo, NOT WITH grep. `grep -c '<joint '`
on the reference says 60 and `mjModel.njnt` says 57; `grep -c '<motor '` says
57 and `nu` says 56; `<geom` says 52 and `ngeom` says 50; `<site` says 6 and
`nsite` says 5. Every difference is an element sitting inside a `<default>`
block. The first draft of this port sized three comptime caps off those greps.

Measured against MuJoCo 3.10.0:
    nq 63   nv 62   nu 56   na 0    nbody 32   nsite 5   nexclude 5
    njnt 57 (1 free + 56 hinge)
    ngeom 50 (1 plane, 8 sphere, 39 capsule, 2 ellipsoid)

Body order is the tree DFS in both engines, so our indices match MuJoCo's here
— asserted in the parity test rather than assumed.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_dims import (
    DM_HUMANOID_CMU_DIMS,
)


comptime _humanoid_cmu_body = """
  <statistic extent="2" center="0 0 1"/>

  <default class="main">
    <joint limited="true" solimplimit="0 0.99 0.01" stiffness="0.1" armature=".01" damping="1"/>
    <geom friction="0.7" solref="0.015 1" solimp="0.95 0.99 0.003"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
    <default class="humanoid">
      <geom type="capsule" material="self"/>
      <default class="stiff_low">
        <joint stiffness=".5" damping="4"/>
      </default>
      <default class="stiff_medium">
        <joint stiffness="10" damping="5"/>
      </default>
      <default class="stiff_high">
        <joint stiffness="30" damping="10"/>
      </default>
      <default class="touch">
        <site group="3" rgba="0 0 1 .5"/>
      </default>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
    <light name="tracking_light" pos="0 0 7" dir="0 0 -1" mode="trackcom"/>
    <camera name="back" pos="0 3 2.4" xyaxes="-1 0 0 0 -1 2" mode="trackcom"/>
    <camera name="side" pos="-3 0 2.4" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
    <body name="root" childclass="humanoid" pos="0 0 1" euler="90 0 0">
      <site name="root" size=".01" rgba="0.5 0.5 0.5 0"/>
      <freejoint name="root"/>
      <geom name="root_geom" size="0.09 0.06" pos="0 -0.05 0" quat="1 0 -1 0"/>
      <body name="lhipjoint">
        <geom name="lhipjoint" size="0.008 0.022" pos="0.051 -0.046 0.025" quat="0.5708 -0.566602 -0.594264 0"/>
        <body name="lfemur" pos="0.102 -0.092 0.05" quat="1 0 0 0.17365">
          <joint name="lfemurrz" axis="0 0 1" range="-60 70" class="stiff_medium"/>
          <joint name="lfemurry" axis="0 1 0" range="-70 70" class="stiff_medium"/>
          <joint name="lfemurrx" axis="1 0 0" range="-160 20" class="stiff_medium"/>
          <geom name="lfemur" size="0.06 0.17" pos="-.01 -0.202473 0" quat="0.7 -0.7 -0.1228 -0.07"/>
          <body name="ltibia" pos="0 -0.404945 0">
            <joint name="ltibiarx" axis="1 0 0" range="1   170" class="stiff_low"/>
            <geom name="ltibia" size="0.03 0.1825614" pos="0 -0.202846 0" quat="0.7 -0.7 -0.1228 -0.1228"/>
            <geom name="lcalf" size="0.045 0.08" pos="0 -0.1 -.01" quat="0.7 -0.7 -0.1228 -0.1228"/>
            <body name="lfoot" pos="0 -0.405693 0" quat="0.707107 -0.707107 0 0">
              <site name="lfoot_touch" type="box" pos="-.005 -.02 -0.025" size=".04 .08 .02" euler="10 0 0" class="touch"/>
              <joint name="lfootrz" axis="0 0 1" range="-70 20" class="stiff_medium"/>
              <joint name="lfootrx" axis="1 0 0" range="-45    90" class="stiff_medium"/>
              <geom name="lfoot0" size="0.02 0.06" pos="-0.02 -0.023 -0.01" euler="100 -2 0"/>
              <geom name="lfoot1" size="0.02 0.06" pos="0 -0.023 -0.01" euler="100 0 0"/>
              <geom name="lfoot2" size="0.02 0.06" pos=".01 -0.023 -0.01" euler="100 10 0"/>
              <body name="ltoes" pos="0 -0.106372 -0.0227756">
                <joint name="ltoesrx" axis="1 0 0" range="-90 20"/>
                <geom name="ltoes0" type="sphere" size="0.02" pos="-.025 -0.01 -.01"/>
                <geom name="ltoes1" type="sphere" size="0.02" pos="0 -0.005 -.01"/>
                <geom name="ltoes2" type="sphere" size="0.02" pos=".02 .001 -.01"/>
                <site name="ltoes_touch" type="capsule" pos="-.005 -.005 -.01" size="0.025 0.02" zaxis="1 .2 0" class="touch"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="rhipjoint">
        <geom name="rhipjoint" size="0.008 0.022" pos="-0.051 -0.046 0.025" quat="0.574856 -0.547594 0.608014 0"/>
        <body name="rfemur" pos="-0.102 -0.092 0.05" quat="1 0 0 -0.17365">
          <joint name="rfemurrz" axis="0 0 1" range="-70 60" class="stiff_medium"/>
          <joint name="rfemurry" axis="0 1 0" range="-70 70" class="stiff_medium"/>
          <joint name="rfemurrx" axis="1 0 0" range="-160 20" class="stiff_medium"/>
          <geom name="rfemur" size="0.06 0.17" pos=".01 -0.202473 0" quat="0.7 -0.7 0.1228 0.07"/>
          <body name="rtibia" pos="0 -0.404945 0">
            <joint name="rtibiarx" axis="1 0 0" range="1   170" class="stiff_low"/>
            <geom name="rtibia" size="0.03 0.1825614" pos="0 -0.202846 0" quat="0.7 -0.7 0.1228 0.1228"/>
            <geom name="rcalf" size="0.045 0.08" pos="0 -0.1 -.01" quat="0.7 -0.7 -0.1228 -0.1228"/>
            <body name="rfoot" pos="0 -0.405693 0" quat="0.707107 -0.707107 0 0">
              <site name="rfoot_touch" type="box" pos=".005 -.02 -0.025" size=".04 .08 .02" euler="10 0 0" class="touch"/>
              <joint name="rfootrz" axis="0 0 1" range="-20 70" class="stiff_medium"/>
              <joint name="rfootrx" axis="1 0 0" range="-45    90" class="stiff_medium"/>
              <geom name="rfoot0" size="0.02 0.06" pos="0.02 -0.023 -0.01" euler="100 2 0"/>
              <geom name="rfoot1" size="0.02 0.06" pos="0 -0.023 -0.01" euler="100 0 0"/>
              <geom name="rfoot2" size="0.02 0.06" pos="-.01 -0.023 -0.01" euler="100 -10 0"/>
              <body name="rtoes" pos="0 -0.106372 -0.0227756">
                <joint name="rtoesrx" axis="1 0 0" range="-90 20"/>
                <geom name="rtoes0" type="sphere" size="0.02" pos=".025 -0.01 -.01"/>
                <geom name="rtoes1" type="sphere" size="0.02" pos="0 -0.005 -.01"/>
                <geom name="rtoes2" type="sphere" size="0.02" pos="-.02 .001 -.01"/>
                <site name="rtoes_touch" type="capsule" pos=".005 -.005 -.01" size="0.025 0.02" zaxis="1 -.2 0" class="touch"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="lowerback">
        <joint name="lowerbackrz" axis="0 0 1" range="-30 30" class="stiff_high"/>
        <joint name="lowerbackry" axis="0 1 0" range="-30 30" class="stiff_high"/>
        <joint name="lowerbackrx" axis="1 0 0" range="-20 45" class="stiff_high"/>
        <geom name="lowerback" size="0.065 0.055" pos="0 0.056 .03" quat="1 0 1 0"/>
        <body name="upperback" pos="0 0.1 -0.01">
          <joint name="upperbackrz" axis="0 0 1" range="-30 30" class="stiff_high"/>
          <joint name="upperbackry" axis="0 1 0" range="-30 30" class="stiff_high"/>
          <joint name="upperbackrx" axis="1 0 0" range="-20 45" class="stiff_high"/>
          <geom name="upperback" size="0.06 0.06" pos="0 0.06 0.02" quat="1 0 1 0"/>
          <body name="thorax" pos="0.000512528 0.11356 0.000936821">
            <joint name="thoraxrz" axis="0 0 1" range="-30 30" class="stiff_high"/>
            <joint name="thoraxry" axis="0 1 0" range="-30 30" class="stiff_high"/>
            <joint name="thoraxrx" axis="1 0 0" range="-20 45" class="stiff_high"/>
            <geom name="thorax" size="0.08 0.07" pos="0 0.05 0" quat="1 0 1 0"/>
            <body name="lowerneck" pos="0 0.113945 0.00468037">
              <joint name="lowerneckrz" axis="0 0 1" range="-30 30" class="stiff_medium"/>
              <joint name="lowerneckry" axis="0 1 0" range="-30 30" class="stiff_medium"/>
              <joint name="lowerneckrx" axis="1 0 0" range="-20 45" class="stiff_medium"/>
              <geom name="lowerneck" size="0.08 0.02" pos="0 0.04 -.02" quat="1 1 0 0"/>
              <body name="upperneck" pos="0 0.09 0.01">
                <joint name="upperneckrz" axis="0 0 1" range="-30 30" class="stiff_medium"/>
                <joint name="upperneckry" axis="0 1 0" range="-30 30" class="stiff_medium"/>
                <joint name="upperneckrx" axis="1 0 0" range="-20 45" class="stiff_medium"/>
                <geom name="upperneck" size="0.05 0.03" pos="0 0.05 0" quat=".8 1 0 0"/>
                <body name="head" pos="0 0.09 0">
                  <camera name="egocentric" pos="0 0 0" xyaxes="-1 0 0 0 1 0" fovy="80"/>
                  <joint name="headrz" axis="0 0 1" range="-30 30" class="stiff_medium"/>
                  <joint name="headry" axis="0 1 0" range="-30 30" class="stiff_medium"/>
                  <joint name="headrx" axis="1 0 0" range="-20 45" class="stiff_medium"/>
                  <geom name="head" size="0.085 0.035" pos="0 0.11 0.03" quat="1 .9 0 0"/>
                  <geom name="leye" type="sphere" size="0.02" pos=" .03 0.11 0.1"/>
                  <geom name="reye" type="sphere" size="0.02" pos="-.03 0.11 0.1"/>
                </body>
              </body>
            </body>
            <body name="lclavicle" pos="0 0.113945 0.00468037">
              <joint name="lclaviclerz" axis="0 0 1" range="0 20" class="stiff_high"/>
              <joint name="lclaviclery" axis="0 1 0" range="-20 10" class="stiff_high"/>
              <geom name="lclavicle" size="0.08 0.04" pos="0.09 0.05 -.01" quat="1 0 -1 -.4"/>
              <body name="lhumerus" pos="0.183 0.076 0.01" quat="0.18 0.68 -0.68 0.18">
                <joint name="lhumerusrz" axis="0 0 1" range="-90 90" class="stiff_low"/>
                <joint name="lhumerusry" axis="0 1 0" range="-90 90" class="stiff_low"/>
                <joint name="lhumerusrx" axis="1 0 0" range="-60 90" class="stiff_low"/>
                <geom name="lhumerus" size="0.035 0.124" pos="0 -0.138 0" quat="0.612 -0.612 0.35 0.35"/>
                <body name="lradius" pos="0 -0.277 0">
                  <joint name="lradiusrx" axis="1 0 0" range="-10 170" class="stiff_low"/>
                  <geom name="lradius" size="0.03 0.06" pos="0 -0.08 0" quat="0.612 -0.612 0.35 0.35"/>
                  <body name="lwrist" pos="0 -0.17 0" quat="-0.5 0 0.866 0">
                    <joint name="lwristry" axis="0 1 0" range="-180 0"/>
                    <geom name="lwrist" size="0.025 0.03" pos="0 -0.02 0" quat="0 0 -1 -1"/>
                    <body name="lhand" pos="0 -0.08 0">
                      <joint name="lhandrz" axis="0 0 1" range="-45 45"/>
                      <joint name="lhandrx" axis="1 0 0" range="-90 90"/>
                      <geom name="lhand" type="ellipsoid" size=".048 0.02 0.06" pos="0 -0.047 0" quat="0 0 -1 -1"/>
                      <body name="lfingers" pos="0 -0.08 0">
                        <joint name="lfingersrx" axis="1 0 0" range="0 90"/>
                        <geom name="lfinger0" size="0.01 0.04" pos="-.03 -0.05 0" quat="1 -1 0 0" />
                        <geom name="lfinger1" size="0.01 0.04" pos="-.008 -0.06 0" quat="1 -1 0 0" />
                        <geom name="lfinger2" size="0.009 0.04" pos=".014 -0.06 0" quat="1 -1 0 0" />
                        <geom name="lfinger3" size="0.008 0.04" pos=".032 -0.05 0" quat="1 -1 0 0" />
                      </body>
                      <body name="lthumb" pos="-.02 -.03 0" quat="0.92388 0 0 -0.382683">
                        <joint name="lthumbrz" axis="0 0 1" range="-45 45"/>
                        <joint name="lthumbrx" axis="1 0 0" range="0 90"/>
                        <geom name="lthumb" size="0.012 0.04" pos="0 -0.06 0" quat="0 0 -1 -1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
            <body name="rclavicle" pos="0 0.113945 0.00468037">
              <joint name="rclaviclerz" axis="0 0 1" range="-20 0" class="stiff_high"/>
              <joint name="rclaviclery" axis="0 1 0" range="-10 20" class="stiff_high"/>
              <geom name="rclavicle" size="0.08 0.04" pos="-.09 0.05 -.01" quat="1 0 -1 .4"/>
              <body name="rhumerus" pos="-0.183 0.076 0.01" quat="0.18 0.68 0.68 -0.18">
                <joint name="rhumerusrz" axis="0 0 1" range="-90 90" class="stiff_low"/>
                <joint name="rhumerusry" axis="0 1 0" range="-90 90" class="stiff_low"/>
                <joint name="rhumerusrx" axis="1 0 0" range="-60 90" class="stiff_low"/>
                <geom name="rhumerus" size="0.035 0.124" pos="0 -0.138 0" quat="0.61 -0.61 -0.35 -0.35"/>
                <body name="rradius" pos="0 -0.277 0">
                  <joint name="rradiusrx" axis="1 0 0" range="-10 170" class="stiff_low"/>
                  <geom name="rradius" size="0.03 0.06" pos="0 -0.08 0" quat="0.612 -0.612 -0.35 -0.35"/>
                  <body name="rwrist" pos="0 -0.17 0" quat="-0.5 0 -0.866 0">
                    <joint name="rwristry" axis="0 1 0" range="-180 0"/>
                    <geom name="rwrist" size="0.025 0.03" pos="0 -0.02 0" quat="0 0 1 1"/>
                    <body name="rhand" pos="0 -0.08 0">
                      <joint name="rhandrz" axis="0 0 1" range="-45 45"/>
                      <joint name="rhandrx" axis="1 0 0" range="-90 90"/>
                      <geom name="rhand" type="ellipsoid" size=".048 0.02 .06" pos="0 -0.047 0" quat="0 0 1 1"/>
                      <body name="rfingers" pos="0 -0.08 0">
                        <joint name="rfingersrx" axis="1 0 0" range="0 90"/>
                        <geom name="rfinger0" size="0.01 0.04" pos=".03 -0.05 0" quat="1 -1  0 0" />
                        <geom name="rfinger1" size="0.01 0.04" pos=".008 -0.06 0" quat="1 -1  0 0" />
                        <geom name="rfinger2" size="0.009 0.04" pos="-.014 -0.06 0" quat="1 -1  0 0" />
                        <geom name="rfinger3" size="0.008 0.04" pos="-.032 -0.05 0" quat="1 -1  0 0" />
                      </body>
                      <body name="rthumb" pos=".02 -.03 0" quat="0.92388 0 0 0.382683">
                        <joint name="rthumbrz" axis="0 0 1" range="-45    45"/>
                        <joint name="rthumbrx" axis="1 0 0" range="0 90"/>
                        <geom name="rthumb" size="0.012 0.04" pos="0 -0.06 0" quat="0 0 1 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="lclavicle" body2="rclavicle"/>
    <exclude body1="lowerneck" body2="lclavicle"/>
    <exclude body1="lowerneck" body2="rclavicle"/>
    <exclude body1="upperneck" body2="lclavicle"/>
    <exclude body1="upperneck" body2="rclavicle"/>
  </contact>

  <actuator>
    <motor name="headrx" joint="headrx" gear="20"/>
    <motor name="headry" joint="headry" gear="20"/>
    <motor name="headrz" joint="headrz" gear="20"/>
    <motor name="lclaviclery" joint="lclaviclery" gear="20"/>
    <motor name="lclaviclerz" joint="lclaviclerz" gear="20"/>
    <motor name="lfemurrx" joint="lfemurrx" gear="120"/>
    <motor name="lfemurry" joint="lfemurry" gear="40"/>
    <motor name="lfemurrz" joint="lfemurrz" gear="40"/>
    <motor name="lfingersrx" joint="lfingersrx" gear="20"/>
    <motor name="lfootrx" joint="lfootrx" gear="20"/>
    <motor name="lfootrz" joint="lfootrz" gear="20"/>
    <motor name="lhandrx" joint="lhandrx" gear="20"/>
    <motor name="lhandrz" joint="lhandrz" gear="20"/>
    <motor name="lhumerusrx" joint="lhumerusrx" gear="40"/>
    <motor name="lhumerusry" joint="lhumerusry" gear="40"/>
    <motor name="lhumerusrz" joint="lhumerusrz" gear="40"/>
    <motor name="lowerbackrx" joint="lowerbackrx" gear="40"/>
    <motor name="lowerbackry" joint="lowerbackry" gear="40"/>
    <motor name="lowerbackrz" joint="lowerbackrz" gear="40"/>
    <motor name="lowerneckrx" joint="lowerneckrx" gear="20"/>
    <motor name="lowerneckry" joint="lowerneckry" gear="20"/>
    <motor name="lowerneckrz" joint="lowerneckrz" gear="20"/>
    <motor name="lradiusrx" joint="lradiusrx" gear="40"/>
    <motor name="lthumbrx" joint="lthumbrx" gear="20"/>
    <motor name="lthumbrz" joint="lthumbrz" gear="20"/>
    <motor name="ltibiarx" joint="ltibiarx" gear="80"/>
    <motor name="ltoesrx" joint="ltoesrx" gear="20"/>
    <motor name="lwristry" joint="lwristry" gear="20"/>
    <motor name="rclaviclery" joint="rclaviclery" gear="20"/>
    <motor name="rclaviclerz" joint="rclaviclerz" gear="20"/>
    <motor name="rfemurrx" joint="rfemurrx" gear="120"/>
    <motor name="rfemurry" joint="rfemurry" gear="40"/>
    <motor name="rfemurrz" joint="rfemurrz" gear="40"/>
    <motor name="rfingersrx" joint="rfingersrx" gear="20"/>
    <motor name="rfootrx" joint="rfootrx" gear="20"/>
    <motor name="rfootrz" joint="rfootrz" gear="20"/>
    <motor name="rhandrx" joint="rhandrx" gear="20"/>
    <motor name="rhandrz" joint="rhandrz" gear="20"/>
    <motor name="rhumerusrx" joint="rhumerusrx" gear="40"/>
    <motor name="rhumerusry" joint="rhumerusry" gear="40"/>
    <motor name="rhumerusrz" joint="rhumerusrz" gear="40"/>
    <motor name="rradiusrx" joint="rradiusrx" gear="40"/>
    <motor name="rthumbrx" joint="rthumbrx" gear="20"/>
    <motor name="rthumbrz" joint="rthumbrz" gear="20"/>
    <motor name="rtibiarx" joint="rtibiarx" gear="80"/>
    <motor name="rtoesrx" joint="rtoesrx" gear="20"/>
    <motor name="rwristry" joint="rwristry" gear="20"/>
    <motor name="thoraxrx" joint="thoraxrx" gear="40"/>
    <motor name="thoraxry" joint="thoraxry" gear="40"/>
    <motor name="thoraxrz" joint="thoraxrz" gear="40"/>
    <motor name="upperbackrx" joint="upperbackrx" gear="40"/>
    <motor name="upperbackry" joint="upperbackry" gear="40"/>
    <motor name="upperbackrz" joint="upperbackrz" gear="40"/>
    <motor name="upperneckrx" joint="upperneckrx" gear="20"/>
    <motor name="upperneckry" joint="upperneckry" gear="20"/>
    <motor name="upperneckrz" joint="upperneckrz" gear="20"/>
  </actuator>
"""


comptime dm_humanoid_cmu_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _humanoid_cmu_body
)

comptime pmhc = DM_HUMANOID_CMU_DIMS

# observation = joint_angles (nq-7 = 56) + head_height (1) + extremities (12)
#             + torso_vertical (3) + com_velocity (3) + velocity (nv = 62)
comptime HUMANOID_CMU_OBS_DIM: Int = 137

comptime DMHumanoidCMUModel = ModelDefFromXML[
    xml=dm_humanoid_cmu_xml,
    xml_path="mojo_rl/envs/dm_control/assets/humanoid_cmu.xml",
    nbody=pmhc.NBODY, njoint=pmhc.NJOINT, nq=pmhc.NQ, nv=pmhc.NV,
    ngeom=pmhc.NGEOM, nact=pmhc.NACT, ntex=pmhc.NTEX, nmat=pmhc.NMAT,
    nlight=pmhc.NLIGHT, ncam=pmhc.NCAM, nsite=pmhc.NSITE,
    # `<contact><exclude>` x5. ⚠ THIS PARAMETER DEFAULTS TO 0 AND NOTHING
    # CHECKS IT. Omitting it builds a model with no exclusions at all, which
    # simulates fine and quietly collides the five body pairs MuJoCo never
    # collides (the two clavicles against each other and against both neck
    # segments). The only symptom is a dynamics divergence you would go looking
    # for in the solver. It was omitted in the first draft of this file, and
    # `merge_mjcf` was ALSO dropping the whole `<contact>` section — two
    # independent zeros multiplying to the same silent answer.
    nexclude=pmhc.NEXCLUDE,
    # humanoid uses 32. Raised here because the CMU skeleton has fingers,
    # thumbs and toes — many more small geoms in close proximity — and an
    # undersized bound DROPS contacts silently. The parity test reports
    # MuJoCo's max `ncon` over its rollouts so this number stays evidence-based.
    max_contacts=64,
    obs_dim_override=HUMANOID_CMU_OBS_DIM,
    timestep=pmhc.TIMESTEP,
]

# Body indices — tree DFS, identical to MuJoCo's (asserted in the parity test).
#
# WARNING: THE REFERENCE BODY IS `thorax`, NOT `torso`. humanoid_CMU has no
# body named torso at all, and `humanoid_CMU.py`'s `torso_vertical_orientation`
# reads the THORAX. Reusing `humanoid`'s TORSO_BODY_IDX = 1 would silently read
# the free-jointed ROOT body instead.
comptime THORAX_BODY_IDX: Int = 14
comptime HEAD_BODY_IDX: Int = 17
comptime LEFT_HAND_BODY_IDX: Int = 22
comptime LEFT_FOOT_BODY_IDX: Int = 5
comptime RIGHT_HAND_BODY_IDX: Int = 29
comptime RIGHT_FOOT_BODY_IDX: Int = 10

comptime N_EXTREMITIES: Int = 4


def extremity_body_indices() -> List[Int]:
    """Bodies whose egocentric offsets form `Physics.extremities()`, IN ORDER.

    The reference iterates `for side in ('l', 'r')` then
    `for limb in ('hand', 'foot')`, so the observation order is lhand, lfoot,
    rhand, rfoot. Getting it wrong permutes 12 observation slots without
    changing the shape — nothing but a value check would catch it.

    WARNING: the SIDE PREFIXES differ from `humanoid`'s ('left_'/'right_') and
    the body names differ; the ORDER happens to coincide. Do not infer one from
    the other.

    A function rather than a `comptime` list: a comptime `List` is not
    `ImplicitlyCopyable`, so it cannot be materialized into a runtime loop.
    """
    return [
        LEFT_HAND_BODY_IDX,
        LEFT_FOOT_BODY_IDX,
        RIGHT_HAND_BODY_IDX,
        RIGHT_FOOT_BODY_IDX,
    ]


# The free root joint occupies qpos[0:7]; `joint_angles()` is qpos[7:].
comptime ROOT_QPOS_SIZE: Int = 7

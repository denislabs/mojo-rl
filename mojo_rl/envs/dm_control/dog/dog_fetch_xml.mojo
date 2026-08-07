"""dm_control `dog fetch` — the model Phase 5 needs (port of `dog.py`).

`fetch` is the one dog task that calls `get_model_and_assets(remove_ball=False)`
and so keeps what stand/walk/trot/run delete: the `ball` body and its
`ball_root` free joint, the `target` geom, four `wall_*` geoms and two cameras.
It also takes `floor_size`'s DEFAULT of 10 rather than
`move_speed * _DEFAULT_TIME_LIMIT`, because `fetch()` overrides only
`remove_ball` — which is why this model is not one of `dog_xml.mojo`'s three.

    stand / walk   floor 15      trot  floor 45      run  floor 135
    fetch          floor 10

⚠ GENERATED, like `dog_xml.mojo`, by `tests/dm_control/dog_ref.py::port_fragment`
— here with `(floor_size=10, remove_ball=False)`. Regenerate rather than
hand-edit; that function is where the port's text-level deviations live.

⚠ SEPARATE FILE, NOT A FOURTH CONSTANT IN `dog_xml.mojo`. The fetch body is its
own ~70 kB of MJCF and dog_xml.mojo is already ~80 kB; §15 of the port plan
measured comptime XML as the dominant build cost, so folding this in would tax
stand/walk/trot/run — which never use it — on every build.

THE `tennis_ball` DEVIATION, and why it is here rather than in the generator
----------------------------------------------------------------------------
dog.xml dresses the ball with

    <texture name="tennis_ball" file="tennis_ball.png" gridsize="3 4" .../>
    <material name="tennis_ball" texture="tennis_ball"/>

`port_fragment` strips both, because they name a PNG ON DISK and a ported XML
carries no asset bundle — MuJoCo cannot compile the string with them present.
That was free for stand/walk/trot/run, which delete the ball; fetch keeps it,
and its geom still says `material="tennis_ball"`. So the material is
re-supplied below as a FLAT COLOUR.

⚠ THIS CHANGES `mat_rgba` AND THE BALL'S `geom_rgba`, and nothing else. It is
rendering-only — no MuJoCo table that feeds the dynamics reads a material — but
it IS a real difference from the reference model, so the parity gate must
exempt those two columns EXPLICITLY rather than pass by luck.

⚠ VERIFIED TO COMPILE AND TO MATCH THE REFERENCE DIMENSIONS before this file
was written (nbody 63, njnt 75, nq 87, nv 85, ngeom 134, nsite 12, nu 38,
ntendon 8), by building the same text with these assets in MuJoCo directly.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_skybox_xml, dm_visual_xml, dm_materials_xml
from .dog_xml import _DOG_SKIN_ASSETS, DOG_FRAME_SKIP, DOG_MAX_STEPS


# The flat-colour stand-in for dog.xml's PNG-textured tennis ball. Merged as a
# separate asset fragment so the body text below stays byte-for-byte what the
# generator emits.
comptime _DOG_FETCH_ASSETS = """<mujoco>
  <asset>
    <material name="tennis_ball" rgba="0.8 0.9 0.25 1" />
  </asset>
</mujoco>"""


comptime _DOG_FETCH_BODY = """<mujoco model="dog">
  <option timestep="0.005" noslip_iterations="4" />
  <size njmax="900" nconmax="300" nkey="1" />
  <default>
    <joint limited="true" solreflimit="0.01 1" solimplimit="0.9 0.99 0.01" stiffness="0.1" armature="0.0001" damping="0.01" />
    <geom friction="0.9" solref="0.01 1" solimp="0.95 0.99 0.001" />
    <tendon rgba="0.5 0 0 1" />
    <general ctrllimited="true" ctrlrange="-1 1" dyntype="filter" dynprm="0.05" gainprm="0.02" />
    <default class="bone">
      <geom type="mesh" contype="0" conaffinity="0" group="5" rgba="0.5 0.5 0.5 1" density="1100.0" />
      <default class="light_bone">
        <geom density="300.0" />
      </default>
      <default class="visible_bone">
        <geom contype="0" conaffinity="0" group="1" />
      </default>
    </default>
    <default class="collision_primitive">
      <geom type="capsule" contype="1" conaffinity="1" condim="1" group="3" rgba="0 0.5 0.5 1" density="300.0" />
      <default class="foot_primitive">
        <geom rgba="0 0.6 0.4 1" solimp="0.9 0.95 0.001" />
      </default>
      <default class="tooth_primitive">
        <geom condim="6" priority="2" friction="0.5 0.01 0.01" />
      </default>
      <default class="nonself_collision_primitive">
        <geom conaffinity="0" />
      </default>
    </default>
    <default class="connector">
      <site group="5" rgba="0 0.5 0 1" size="0.005" />
    </default>
    <default class="muscle">
      <geom contype="0" conaffinity="0" group="4" rgba="0.5 0 0 1" />
    </default>
    <default class="lumbar">
      <joint group="3" pos="0.012 0 0.01" armature="0.01" damping="3.0" />
      <general gainprm="40" />
      <default class="lumbar_extend">
        <joint stiffness="30.0" range="-24.5 24.5" />
        <general gainprm="60" />
      </default>
      <default class="lumbar_bend">
        <joint stiffness="45.0" range="-21 21" />
      </default>
      <default class="lumbar_twist">
        <joint stiffness="45.0" range="-16.333 16.333" />
      </default>
    </default>
    <default class="cervical">
      <joint group="3" pos="-0.7 0 0" stiffness="4.0" armature="0.01" damping="3.0" />
      <default class="cervical_extend">
        <joint range="-17.5 61.25" />
        <general gainprm="20" />
      </default>
      <default class="cervical_bend">
        <joint range="-35 35" />
        <general gainprm="20" />
      </default>
      <default class="cervical_twist">
        <joint range="-23.333 23.333" />
        <general gainprm="14" />
      </default>
    </default>
    <default class="atlas">
      <joint axis="0 -1 0" stiffness="2.0" range="-15 30" damping="0.2" />
      <general gainprm="10" />
    </default>
    <default class="caudal">
      <joint group="3" springdamper="0.001 50" />
      <general gainprm="0.5" />
      <default class="caudal_extend">
        <joint range="-15 15" />
      </default>
      <default class="caudal_bend">
        <joint range="-15 15" />
      </default>
    </default>
    <default class="mandible">
      <joint stiffness="2.0" range="-15 20" springref="-11.0" damping="0.2" />
      <general gainprm="3" />
    </default>
    <default class="hip">
      <joint stiffness="5.0" armature="0.01" damping="0.5" />
      <general gainprm="10" />
      <default class="hip_supinate">
        <joint range="-20 20" />
      </default>
      <default class="hip_abduct">
        <joint range="-30 90" />
      </default>
      <default class="hip_extend">
        <joint range="-130 60" />
        <general gainprm="40" />
      </default>
    </default>
    <default class="knee">
      <joint range="-120 10" damping="0.5" />
      <general gainprm="30" />
    </default>
    <default class="ankle">
      <joint range="-100 20" damping="0.3" />
      <general gainprm="20" />
    </default>
    <default class="toe">
      <joint stiffness="2.0" range="-15 70" damping="0.02" />
      <general gainprm="2" />
    </default>
    <default class="scapula">
      <joint stiffness="10.0" damping="1.5" />
      <general gainprm="30" />
      <default class="scapula_supinate">
        <joint pos="-0.03 0.02 0" range="-15 35" />
      </default>
      <default class="scapula_abduct">
        <joint pos="0 0.05 -0.07" range="-15 20" />
      </default>
      <default class="scapula_extend">
        <joint pos="0 0 0" range="-30 30" />
        <general gainprm="30" />
      </default>
    </default>
    <default class="shoulder">
      <joint stiffness="5.0" damping="1.0" />
      <general gainprm="30" />
      <default class="shoulder_extend">
        <joint range="-40 60" />
      </default>
      <default class="shoulder_supinate">
        <joint range="-30 15" />
      </default>
    </default>
    <default class="elbow">
      <joint range="-100 30" damping="0.3" />
      <general gainprm="20" />
    </default>
    <default class="wrist">
      <joint range="-110 10" damping="0.2" />
      <general gainprm="10" />
    </default>
    <default class="finger">
      <joint stiffness="1.0" range="-15 100" damping="0.02" />
      <general gainprm="2" />
    </default>
    <default class="sensor">
      <site group="4" />
    </default>
    <default class="bouncy">
      <geom condim="6" priority="1" friction="0.5 0.001 0.001" solref="-5000 -5" solimp="0.5 0.9 0.01" />
    </default>
    <default class="velcro">
      <geom condim="6" priority="2" friction="0.5 0.01 0.01" />
    </default>
  </default>
  <visual>
    <global offwidth="1920" offheight="1080" />
    <map znear="0.001" />
    <scale jointlength="8.0" jointwidth="0.5" framelength="2.0" framewidth="0.3" />
  </visual>
  <statistic meansize="0.01" extent="1.0" />
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1" material="grid" />
    <body name="torso" pos="0.040403 0 0.41522">
      <inertial pos="-1.5426e-07 0.0 2.4738e-07" quat="0.45516936012336334 0.5411292392810502 0.5411292392810502 0.45516936012336334" mass="2.6757659178332736" diaginertia="0.026026 0.023326 0.011595" /><freejoint name="root" />
      <light name="light" mode="trackcom" pos="0 0 3" />
      <camera name="y-axis" mode="trackcom" pos="0 -1.5 0.8" xyaxes="1 0 0 0 0.6 1" />
      <camera name="x-axis" mode="trackcom" pos="2 0 0.5" xyaxes="0 1 0 -0.3 0 1" />
      <geom name="collision_torso" class="nonself_collision_primitive" type="ellipsoid" size="0.2 0.09 0.11" density="200.0" pos="0 0 0" euler="0 10 0" />
      <body name="L_1" pos="-0.1235 0 0.09287">
        <inertial pos="3.939961677114705e-09 1.17396449944456e-10 -0.018551757579928285" quat="0.707097086107865 0.0037028121431095682 0.0037028121431095682 0.707097086107865" mass="0.16934158036957048" diaginertia="0.00016280910250044805 0.00016278127992222375 0.00015805562621924913" /><geom name="L_1_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
        <body name="L_2" pos="-0.023024 0 0.00096026">
          <inertial pos="1.351412831168936e-07 -3.9918148095170206e-11 -0.01852941728517009" quat="0.7070940968793715 0.004235346307754391 0.004235346307754391 0.7070940968793715" mass="0.16954556927292408" diaginertia="0.00016293049375907734 0.00016292389236086693 0.00015811238663932293" /><geom name="L_2_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
          <body name="L_3" pos="-0.0233 0 0.00039242">
            <inertial pos="1.4498136069046815e-07 -2.0228643743727128e-10 -0.018500085733431697" quat="0.9999974003164869 0.0 0.0022802105753499786 0.0" mass="0.16981270195851367" diaginertia="0.00016315407927636812 0.00016310157792592142 0.00015817394860825177" /><geom name="L_3_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
            <body name="L_4" pos="-0.024246 0 -0.00041778">
              <inertial pos="-3.424200313596861e-07 -4.632706889500715e-10 -0.018366060414586823" quat="0.9999891409798439 0.0 -0.004660249177234591 0.0" mass="0.17105759407286236" diaginertia="0.00016385576092853185 0.0001637943193058452 0.00015852017191419912" /><geom name="L_4_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
              <body name="L_5" pos="-0.024458 0 -0.00014988">
                <inertial pos="-1.7014528812726014e-07 -1.2675630844095226e-09 -0.018382053206141506" quat="0.999886727161681 0.0 -0.015051008135738687 0.0" mass="0.17090864545716766" diaginertia="0.0001639956147926749 0.00016377786927536102 0.00015858432286329415" /><geom name="L_5_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
                <body name="L_6" pos="-0.023445 0 0.00066898">
                  <inertial pos="2.441984954394653e-07 -1.424035205917774e-09 -0.01837621875805433" quat="0.999861744209388 0.0 -0.01662806262197623 0.0" mass="0.17095614049320254" diaginertia="0.00016401927263038263 0.0001637822978324267 0.00015860659419601923" /><geom name="L_6_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
                  <body name="L_7" pos="-0.026399 0 0.0014219">
                    <inertial pos="4.0298195374310424e-07 -1.8426302148790338e-09 -0.018142013701116285" quat="0.9998094750892872 0.0 -0.019519567661300044 0.0" mass="0.17316967888977" diaginertia="0.0001649388550018783 0.0001647351571643661 0.00015915109817302985" /><geom name="L_7_collision" class="nonself_collision_primitive" type="sphere" size="0.05" pos="0 0 -0.02" />
                    <joint name="L_7_extend" class="lumbar_extend" axis="0 1 0" />
                    <joint name="L_7_bend" class="lumbar_bend" axis="0 0 1" />
                    <body name="pelvis" pos="-0.080217 0 -0.034801">
                      <inertial pos="0.008503582633938543 -1.0427252339321508e-08 -0.005507541821947402" quat="-3.561184841752615e-18 0.5563149694890251 -2.1502293361067585e-17 0.8309715125817666" mass="1.155341815462466" diaginertia="0.0035593208381731016 0.0031411674708173632 0.0015244920729198532" /><geom name="collision_pelvis_L" class="nonself_collision_primitive" size="0.05 0.05 0" pos="0.01 0.02 -0.01" euler="0 70 0" />
                      <geom name="collision_pelvis_R" class="nonself_collision_primitive" size="0.05 0.05 0" pos="0.01 -0.02 -0.01" euler="0 70 0" />
                      <body name="upper_leg_L" pos="-0.018108 0.047237 -0.012597">
                        <inertial pos="-0.016011216036232588 0.01476251273880773 -0.07381274938599021" quat="0.5368788478139312 0.12997714710202987 -0.03219316432663039 0.8329649717556213" mass="0.6559920686856067" diaginertia="0.0025507239686046435 0.0023982419419068344 0.0005778445809755943" /><joint name="hip_L_supinate" class="hip_supinate" axis="0 0 1" />
                        <joint name="hip_L_abduct" class="hip_abduct" axis="1 0 0" />
                        <joint name="hip_L_extend" class="hip_extend" axis="0 1 0" />
                        <geom name="upper_leg_L0_collision" class="collision_primitive" size="0.04 0.08" pos="-0.01 0.02 -0.08" euler="10 20 0" />
                        <geom name="upper_leg_L1_collision" class="collision_primitive" size="0.04 0.04" pos="-0.03 0 -0.05" euler="10 5 0" />
                        <body name="lower_leg_L" pos="-0.023037 0.0311 -0.16702">
                          <inertial pos="-0.082330302496435 4.68488703007003e-05 -0.05933809007605692" quat="0.8027481045818837 0.1850459424598865 0.42145424237868234 0.3791171340764473" mass="0.20676150764834839" diaginertia="0.0009091512622260609 0.0009056766220318612 3.405283613426215e-05" /><joint name="knee_L" class="knee" axis="0 -1 0" />
                          <body name="foot_L" pos="-0.17914 -0.0056423 -0.13824">
                            <inertial pos="-0.011400928427426417 0.003990654768267846 -0.04399806785847883" quat="0.9660764406172341 0.0069100538661049215 -0.0006162751575999276 0.25816309233667273" mass="0.08545298261343763" diaginertia="0.0001688305104944155 0.0001663174133496424 9.357585981971725e-06" /><joint name="ankle_L" class="ankle" axis="0 1 0" />
                            <geom name="foot_L_collision" class="collision_primitive" size="0.015 0.07" pos="-0.01 0.005 -0.05" />
                            <body name="foot_anchor_L" pos="-0.015043 0.0081311 -0.11993">
                              <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.00030000000000000003" diaginertia="5.000000000000001e-09 5.000000000000001e-09 5.000000000000001e-09" /><geom name="foot_anchor_L" class="foot_primitive" type="box" contype="0" conaffinity="0" size="0.005 0.005 0.005" />
                              <site name="foot_anchor_L" class="sensor" />
                              <body name="toe_L">
                                <inertial pos="0.024543844892396287 -0.00046926779417957783 -0.019860010947774526" quat="0.2708023647967948 0.7994292881265387 -0.12082691447819816 0.522474639809093" mass="0.03210790196974051" diaginertia="1.3219087041565478e-05 9.383497299111521e-06 8.630017571643527e-06" /><joint name="toe_L" class="toe" axis="0 1 0" />
                                <geom name="toe_L0_collision" class="foot_primitive" size="0.018 0.012" pos="0.015 0 -0.02" euler="90 0 0" />
                                <geom name="toe_L1_collision" class="foot_primitive" size="0.01 0.015" pos="0.035 0 -0.028" euler="90 0 0" />
                                <geom name="toe_L2_collision" class="foot_primitive" size="0.008 0.01" pos="0.045 0 -0.03" euler="90 0 0" />
                                <site name="sole_L" class="sensor" type="box" size="0.025 0.03 0.008" pos="0.026 0 -0.033" />
                              </body>
                            </body>
                          </body>
                          <geom name="lower_leg_L_collision" class="collision_primitive" size="0.02 0.1" pos="-0.09513 -1.1454e-05 -0.068643" quat="0.80051 0.18703 0.42091 0.38345" />
                        </body>
                      </body>
                      <body name="upper_leg_R" pos="-0.018108 -0.047237 -0.012597">
                        <inertial pos="-0.01601121598548625 -0.014762524812579886 -0.07381274918172333" quat="0.832965115858225 -0.032193241563394995 0.1299771567889179 0.5368786172627897" mass="0.6559920669804127" diaginertia="0.0025507241031319555 0.0023982420069648567 0.000577844669384969" /><joint name="hip_R_supinate" class="hip_supinate" axis="0 0 -1" />
                        <joint name="hip_R_abduct" class="hip_abduct" axis="-1 0 0" />
                        <joint name="hip_R_extend" class="hip_extend" axis="0 1 0" />
                        <geom name="upper_leg_R0_collision" class="collision_primitive" size="0.04 0.08" pos="-0.01 -0.02 -0.08" euler="-10 20 0" />
                        <geom name="upper_leg_R1_collision" class="collision_primitive" size="0.04 0.04" pos="-0.03 0 -0.05" euler="-10 5 0" />
                        <body name="lower_leg_R" pos="-0.023037 -0.0311 -0.16702">
                          <inertial pos="-0.08233030314133202 -4.693073754609154e-05 -0.05933809088434707" quat="0.3791170278186138 0.4214542441691265 0.18504594152839382 0.8027481540394121" mass="0.2067615052259982" diaginertia="0.000909151236450573 0.0009056765952554193 3.4052836726707565e-05" /><joint name="knee_R" class="knee" axis="0 -1 0" />
                          <body name="foot_R" pos="-0.17914 0.0056423 -0.13824">
                            <inertial pos="-0.011400928453154668 -0.003990710117513978 -0.043998066085631365" quat="0.9660761623152182 -0.006910004970955367 -0.0006162706179328136 -0.2581641350926326" mass="0.08545297520636723" diaginertia="0.00016883049737482253 0.00016631739656095454 9.357578276025655e-06" /><joint name="ankle_R" class="ankle" axis="0 1 0" />
                            <geom name="foot_R_collision" class="collision_primitive" size="0.015 0.07" pos="-0.01 -0.005 -0.05" />
                            <body name="foot_anchor_R" pos="-0.015043 -0.0081311 -0.11993">
                              <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.00030000000000000003" diaginertia="5.000000000000001e-09 5.000000000000001e-09 5.000000000000001e-09" /><geom name="foot_anchor_R" class="foot_primitive" type="box" contype="0" conaffinity="0" size="0.005 0.005 0.005" />
                              <site name="foot_anchor_R" class="sensor" />
                              <body name="toe_R">
                                <inertial pos="0.024543845011702925 0.0004692355365163758 -0.0198600110885127" quat="-0.270800718709639 0.7994298101245058 0.12082678048777531 0.5224747252715699" mass="0.03210790169196016" diaginertia="1.3219083877570437e-05 9.383494996515561e-06 8.630019026703526e-06" /><joint name="toe_R" class="toe" axis="0 1 0" />
                                <geom name="toe_R0_collision" class="foot_primitive" size="0.018 0.012" pos="0.015 0 -0.02" euler="90 0 0" />
                                <geom name="toe_R1_collision" class="foot_primitive" size="0.01 0.015" pos="0.035 0 -0.028" euler="90 0 0" />
                                <geom name="toe_R2_collision" class="foot_primitive" size="0.008 0.01" pos="0.045 0 -0.03" euler="90 0 0" />
                                <site name="sole_R" class="sensor" type="box" size="0.025 0.03 0.008" pos="0.026 0 -0.033" />
                              </body>
                            </body>
                          </body>
                          <geom name="lower_leg_R_collision" class="collision_primitive" size="0.02 0.1" pos="-0.09513 1.136e-05 -0.068643" quat="0.38345 0.42091 0.18703 0.80051" />
                        </body>
                      </body>
                      <body name="Ca_1" pos="0.0082462 0 0.04354">
                        <inertial pos="3.5791810158144796e-06 -7.755014933702828e-09 -3.886041225033732e-06" quat="-1.54125680588483e-17 0.41959953153653384 -1.2735651409218963e-17 0.9077093329553936" mass="0.007790614637162276" diaginertia="7.953164186989729e-07 7.878667479408177e-07 5.133622702098722e-07" /><body name="Ca_2" pos="-0.01491 0 4.6796e-05">
                          <inertial pos="3.1787446840903204e-06 -7.833886100020762e-09 2.9098821639245547e-06" quat="1.9271022002194137e-17 0.5116399999348118 2.9182856169295404e-18 0.8591999246198234" mass="0.006779977515463819" diaginertia="7.482997515438126e-07 7.439732277135909e-07 3.752338305691207e-07" /><body name="Ca_3" pos="-0.01408 0 -0.0010873">
                            <inertial pos="2.806827738997428e-06 -7.3457130926383395e-09 -4.3884841751192e-06" quat="-1.8529914923971255e-17 0.44900876891423547 -2.383526936003071e-18 0.8935273501343576" mass="0.0060139064294942235" diaginertia="7.318266870649172e-07 7.282117123527e-07 2.8980271337447656e-07" /><body name="Ca_4" pos="-0.012508 0 -0.002837">
                              <inertial pos="4.886790723639035e-06 -1.047783028194975e-08 -1.4063763526127203e-06" quat="1.2609083017676585e-17 0.39440798462579313 -1.3297465190457833e-17 0.9189354393337021" mass="0.004242581858782669" diaginertia="4.426908332809764e-07 4.394870747868544e-07 1.3907382994453785e-07" /><body name="Ca_5" pos="-0.013783 -1.652e-06 -0.0065348">
                                <inertial pos="1.929419019773308e-06 1.5368429078800936e-10 3.833222586433381e-06" quat="-0.0002784067577000457 0.5096721553811445 0.0004699735083248825 0.8603685231595364" mass="0.002949456914576409" diaginertia="2.4849046286016053e-07 2.470067013446201e-07 6.188915631486894e-08" /><body name="Ca_6" pos="-0.016384 -1.0009e-05 -0.0098338">
                                  <inertial pos="-1.830758072576797e-06 5.908668342156892e-11 1.975043175155823e-09" quat="-0.00010865046555866528 0.47375045330691545 0.00020197135612821872 0.8806591028281849" mass="0.0028959301556064224" diaginertia="2.308597549452438e-07 2.2976449601415555e-07 5.5437425285346813e-08" /><body name="Ca_7" pos="-0.016206 1.4193e-05 -0.011471">
                                    <inertial pos="1.9004449747534118e-06 -8.156910517848101e-11 -8.769060060284887e-07" quat="0.6266334374733089 0.3277272272692061 0.3276133926450295 0.6265739098513755" mass="0.00263939675172212" diaginertia="1.9535155558507182e-07 1.951063769374564e-07 4.859745439756195e-08" /><body name="Ca_8" pos="-0.018452 0.00021258 -0.013121">
                                      <inertial pos="-3.5968036204585587e-07 -1.7760347115623998e-09 -1.575090395584305e-06" quat="3.284811308146716e-05 0.41307146828850666 -7.242023033797028e-05 0.9106986086309197" mass="0.0014715717163338552" diaginertia="8.416961921555981e-08 8.414810374412363e-08 1.3834448873165167e-08" /><body name="Ca_9" pos="-0.019834 -1.086e-05 -0.017678">
                                        <inertial pos="-4.8467914942329186e-06 -1.4931623237912825e-09 -1.4566909209259848e-08" quat="0.6532885789577236 0.27066730219470947 0.27058091692393854 0.6532527929696429" mass="0.0018738900745092512" diaginertia="1.4168151531610973e-07 1.4164536162144346e-07 2.1669586591335682e-08" /><body name="Ca_10" pos="-0.023389 1.1411e-05 -0.022029">
                                          <inertial pos="-3.8012819536164867e-06 -1.15027461326571e-09 8.01638224350995e-07" quat="-8.487593155415115e-05 0.3978496468917073 0.00019572588634524625 0.9174506051856969" mass="0.0025716139062341567" diaginertia="3.01704699788701e-07 3.0152125743872847e-07 3.1070470936460235e-08" /><body name="Ca_11" pos="-0.024196 -5.3444e-06 -0.022377">
                                            <inertial pos="-4.308735932021449e-08 1.5522511553556438e-09 3.6294637198471366e-06" quat="-3.1852090209219346e-05 0.41645726060478144 6.953533796037486e-05 0.9091552915975611" mass="0.002317407242356073" diaginertia="2.6504582865077797e-07 2.646864118906865e-07 2.566720379548271e-08" /><body name="Ca_12" pos="-0.024081 -5.5423e-05 -0.019678">
                                              <inertial pos="-1.115715288044511e-06 1.1646572859181852e-10 -4.20990338095139e-06" quat="-2.3503775966160408e-05 0.433517670755796 4.885685895607827e-05 0.9011450639065017" mass="0.001958394206104485" diaginertia="2.1802030055324356e-07 2.1788333261948533e-07 1.8673812145222004e-08" /><body name="Ca_13" pos="-0.02323 -1.2027e-05 -0.017776">
                                                <inertial pos="-1.5794406453564073e-06 1.4239469936353391e-09 -1.0868886699696289e-08" quat="0.6343395611128733 0.31246664714497935 0.31243130638130484 0.6343221535008662" mass="0.0014257107989850905" diaginertia="1.283338702739566e-07 1.2829485317217405e-07 1.2309176783135946e-08" /><body name="Ca_14" pos="-0.023245 -9.0352e-05 -0.018326">
                                                  <inertial pos="3.905997375389659e-06 -1.4981078936728335e-10 3.6119902503361047e-06" quat="0.6307176890679552 0.32187961087302946 0.31967983467334876 0.629597900333402" mass="0.0016898815939799185" diaginertia="1.6238653826100391e-07 1.6231807532574777e-07 1.3116778096878695e-08" /><body name="Ca_15" pos="-0.024641 5.041e-05 -0.014254">
                                                    <inertial pos="3.3713252596836094e-06 5.406413482659868e-10 -4.4426675598643146e-07" quat="0.0009847178051356931 0.5613238834371305 -0.0014518300810100215 0.8275943574018151" mass="0.0010140152556605622" diaginertia="7.229592127121792e-08 7.226790473536929e-08 5.8316625062476154e-09" /><body name="Ca_16" pos="-0.023338 -3.1369e-05 -0.0085014">
                                                      <inertial pos="-4.248979930431488e-06 9.358373747214065e-11 -1.8936379462106422e-06" quat="0.0011236766879616593 0.5922622350348122 -0.0015287057255677946 0.8057430392880092" mass="0.0008384694618995164" diaginertia="6.831653965921883e-08 6.82873960573423e-08 4.256326370497432e-09" /><body name="Ca_17" pos="-0.024444 -1.4083e-05 -0.0053673">
                                                        <inertial pos="1.5131243824758433e-06 2.8378576168741174e-10 8.412050350367126e-07" quat="0.0009343616003239924 0.6621974974020962 -0.0010573028921149214 0.7493280213053553" mass="0.0011797479264167203" diaginertia="1.1935491355683594e-07 1.1931829065805844e-07 7.1576099476659156e-09" /><body name="Ca_18" pos="-0.02601 1.3546e-05 -0.0022059">
                                                          <inertial pos="1.2306224602030758e-06 3.394558597147016e-11 4.981334248273652e-06" quat="0.5131803722594626 0.48769136716674905 0.4864626455624737 0.5120128224869263" mass="0.0011383927759692036" diaginertia="1.1050042539165346e-07 1.1041045769830285e-07 6.747676441375106e-09" /><body name="Ca_19" pos="-0.026655 3.2028e-05 0.00068575">
                                                            <inertial pos="-3.6363450982517845e-06 1.8033791604156595e-09 7.32163227639228e-07" quat="0.0010458856886161694 0.7549412328374097 -0.00090852423451187 0.6557909847419208" mass="0.000846128171269107" diaginertia="8.096084720671141e-08 8.09291250757447e-08 3.919221972517309e-09" /><body name="Ca_20" pos="-0.026838 -6.5667e-05 0.0040269">
                                                            <inertial pos="-1.9503416225958426e-06 2.6536748873956407e-10 -2.4149456340107845e-06" quat="6.796917616053096e-05 0.7653454214787639 -5.715890197374573e-05 0.643619746383335" mass="0.0004923675968507898" diaginertia="3.540442792648742e-08 3.538299450808647e-08 1.3483851694419642e-09" /><body name="Ca_21" pos="-0.020294 4.4629e-05 0.0048627">
                                                            <inertial pos="3.744442612504802e-06 -1.3048141687002044e-10 2.826547327398734e-07" quat="0.434163086681847 0.5581240132469596 0.5581240132469596 0.434163086681847" mass="0.00029124278085185327" diaginertia="1.8149447151777514e-08 1.7715455381389313e-08 1.0496171344730112e-09" /><joint name="Ca_21_extend" class="caudal_extend" pos="0.010185 -2.2399e-05 -0.0024405" axis="0 1 0" />
                                                            <site name="tail_tip" class="sensor" size="0.005" />
                                                            <geom name="Ca_21_collision" class="collision_primitive" size="0.0023331 0.014653 0.010474" pos="3.7435e-06 0 2.826e-07" quat="-0.017531 0.80227 -0.023589 0.59624" />
                                                            </body>
                                                            <joint name="Ca_20_bend" class="caudal_bend" pos="0.012076 2.9547e-05 -0.0018119" axis="-0.14838 0 -0.98893" />
                                                            <geom name="Ca_20_collision" class="collision_primitive" size="0.0027962 0.01437 0.012211" pos="-1.9507e-06 0 -2.415e-06" quat="-0.082662 0.75899 -0.098524 0.63828" />
                                                            </body>
                                                            <joint name="Ca_19_extend" class="caudal_extend" pos="0.013735 -1.6504e-05 -0.00035336" axis="0 1 0" />
                                                            <geom name="Ca_19_collision" class="collision_primitive" size="0.0036068 0.016752 0.01374" pos="-3.6363e-06 0 7.3241e-07" quat="-0.064025 0.75105 -0.07658 0.65266" />
                                                          </body>
                                                          <joint name="Ca_18_bend" class="caudal_bend" pos="0.013759 -7.1658e-06 0.0011669" axis="0.084506 0 -0.99642" />
                                                          <geom name="Ca_18_collision" class="collision_primitive" size="0.0041192 0.016385 0.013809" pos="1.2295e-06 0 4.9815e-06" quat="0.51343 0.48795 0.4862 0.51176" />
                                                        </body>
                                                        <joint name="Ca_17_extend" class="caudal_extend" pos="0.012585 7.2505e-06 0.0027633" axis="0 1 0" />
                                                        <geom name="Ca_17_collision" class="collision_primitive" size="0.0041188 0.01702 0.012885" pos="1.5136e-06 0 8.4095e-07" quat="-0.1046 0.65534 -0.095041 0.74199" />
                                                      </body>
                                                      <joint name="Ca_16_bend" class="caudal_bend" pos="0.011122 1.4949e-05 0.0040515" axis="0.34227 0 -0.9396" />
                                                      <geom name="Ca_16_collision" class="collision_primitive" size="0.0037779 0.014767 0.011837" pos="-4.2489e-06 0 -1.8936e-06" quat="-0.085016 0.5886 -0.065817 0.80125" />
                                                    </body>
                                                    <joint name="Ca_15_extend" class="caudal_extend" pos="0.010103 -2.0668e-05 0.005844" axis="0 1 0" />
                                                    <geom name="Ca_15_collision" class="collision_primitive" size="0.0040575 0.013822 0.011671" pos="3.3718e-06 0 -4.4389e-07" quat="-0.083613 0.55813 -0.059816 0.82336" />
                                                  </body>
                                                  <joint name="Ca_14_bend" class="caudal_bend" pos="0.012094 4.7007e-05 0.0095343" axis="0.61911 0 -0.78529" />
                                                  <geom name="Ca_14_collision" class="collision_primitive" size="0.0048495 0.014433 0.0154" pos="3.9059e-06 0 3.612e-06" quat="0.68073 0.29564 0.3441 0.57516" />
                                                </body>
                                                <joint name="Ca_13_extend" class="caudal_extend" pos="0.010237 5.3e-06 0.0078334" axis="0 1 0" />
                                                <geom name="Ca_13_collision" class="collision_primitive" size="0.0050115 0.014033 0.01289" pos="-1.5794e-06 0 0" quat="0.65681 0.301 0.32349 0.61102" />
                                              </body>
                                              <joint name="Ca_12_bend" class="caudal_bend" pos="0.011708 2.6945e-05 0.0095669" axis="0.63276 0 -0.77434" />
                                              <geom name="Ca_12_collision" class="collision_primitive" size="0.0052969 0.01585 0.015119" pos="-1.1154e-06 0 -4.2092e-06" quat="-0.067175 0.43232 -0.032219 0.89864" />
                                            </body>
                                            <joint name="Ca_11_extend" class="caudal_extend" pos="0.011106 2.453e-06 0.010271" axis="0 1 0" />
                                            <geom name="Ca_11_collision" class="collision_primitive" size="0.0056974 0.016024 0.015127" pos="0 0 3.6296e-06" quat="-0.044239 0.41597 -0.020132 0.90808" />
                                          </body>
                                          <joint name="Ca_10_bend" class="caudal_bend" pos="0.012071 -5.8892e-06 0.011369" axis="0.68563 0 -0.72795" />
                                          <geom name="Ca_10_collision" class="collision_primitive" size="0.0060592 0.014866 0.016582" pos="-3.8011e-06 0 8.0203e-07" quat="-0.055024 0.39716 -0.023434 0.9158" />
                                        </body>
                                        <joint name="Ca_9_extend" class="caudal_extend" pos="0.0090216 4.9397e-06 0.008041" axis="0 1 0" />
                                        <geom name="Ca_9_collision" class="collision_primitive" size="0.005861 0.01177 0.012085" pos="-4.8466e-06 0 0" quat="0.7194 0.24021 0.29795 0.57964" />
                                      </body>
                                      <joint name="Ca_8_bend" class="caudal_bend" pos="0.0089967 -0.00010365 0.0063975" axis="0.57949 0 -0.81493" />
                                      <geom name="Ca_8_collision" class="collision_primitive" size="0.0052297 0.010249 0.01104" pos="-3.6054e-07 0 -1.5754e-06" quat="-0.22403 0.40034 -0.10176 0.88271" />
                                    </body>
                                    <joint name="Ca_7_extend" class="caudal_extend" pos="0.0081104 -7.103e-06 0.0057407" axis="0 1 0" />
                                    <geom name="Ca_7_collision" class="collision_primitive" size="0.0072624 0.011349 0.0099365" pos="1.9008e-06 0 -8.7709e-07" quat="0.62321 0.32952 0.32581 0.62998" />
                                  </body>
                                  <joint name="Ca_6_bend" class="caudal_bend" pos="0.0082633 5.0481e-06 0.0049597" axis="0.51463 0 -0.85741" />
                                  <geom name="Ca_6_collision" class="collision_primitive" size="0.0073248 0.012258 0.0096375" pos="-1.8304e-06 0 0" quat="-0.014581 0.47369 -0.007547 0.88054" />
                                </body>
                                <joint name="Ca_5_extend" class="caudal_extend" pos="0.010037 1.203e-06 0.0047586" axis="0 1 0" />
                                <geom name="Ca_5_collision" class="collision_primitive" size="0.007544 0.012356 0.011108" pos="1.9298e-06 0 3.8335e-06" quat="0.0073058 0.50965 0.0050246 0.86034" />
                              </body>
                              <joint name="Ca_4_bend" class="caudal_bend" pos="0.010088 0 0.0022882" axis="0.2212 0 -0.97523" />
                              <geom name="Ca_4_collision" class="collision_primitive" size="0.0092021 0.012461 0.010344" pos="4.8867e-06 0 -1.4063e-06" quat="0 0.39441 0 0.91894" />
                            </body>
                            <joint name="Ca_3_extend" class="caudal_extend" pos="0.01091 0 0.00084249" axis="0 1 0" />
                            <geom name="Ca_3_collision" class="collision_primitive" size="0.011207 0.011436 0.010942" pos="2.8069e-06 0 -4.389e-06" quat="0 0.44901 0 0.89353" />
                          </body>
                          <joint name="Ca_2_bend" class="caudal_bend" pos="0.011646 0 -3.6552e-05" axis="-0.0031385 0 -1" />
                          <geom name="Ca_2_collision" class="collision_primitive" size="0.012209 0.0092225 0.011646" pos="3.1789e-06 0 2.9096e-06" quat="0 0.51164 0 0.8592" />
                        </body>
                        <joint name="Ca_1_extend" class="caudal_extend" pos="0.011214 0 -0.0020078" axis="0 1 0" />
                        <geom name="Ca_1_collision" class="collision_primitive" size="0.013645 0.0064555 0.011392" pos="3.5794e-06 0 -3.8864e-06" quat="0 0.4196 0 0.90771" />
                      </body>
                    </body>
                  </body>
                  <joint name="L_6_twist" class="lumbar_twist" axis="1 0 0" />
                </body>
                <joint name="L_5_extend" class="lumbar_extend" axis="0 1 0" />
                <joint name="L_5_bend" class="lumbar_bend" axis="0 0 1" />
              </body>
              <joint name="L_4_twist" class="lumbar_twist" axis="1 0 0" />
            </body>
            <joint name="L_3_extend" class="lumbar_extend" axis="0 1 0" />
            <joint name="L_3_bend" class="lumbar_bend" axis="0 0 1" />
          </body>
          <joint name="L_2_twist" class="lumbar_twist" axis="1 0 0" />
        </body>
        <joint name="L_1_extend" class="lumbar_extend" axis="0 1 0" />
        <joint name="L_1_bend" class="lumbar_bend" axis="0 0 1" />
      </body>
      <body name="C_7" pos="0.1433 0 0.0869">
        <inertial pos="-1.7347094433985887e-08 1.0336659114747611e-10 2.6451551312348574e-09" quat="0.9765266761750729 0.0 -0.2153964965324741 0.0" mass="0.43381584111654065" diaginertia="0.000845149268312954 0.0008450970405480047 0.0008449767895681831" /><geom name="C_7_R_collision" class="nonself_collision_primitive" type="sphere" size="0.07" />
        <body name="C_6" pos="0.018193 0 0.0093423">
          <inertial pos="-1.7147206391782274e-10 2.9244155088614706e-10 4.529570749966154e-08" quat="0.9751066307294728 0.0 -0.22173646228668756 0.0" mass="0.33497171268124315" diaginertia="0.0005406902921345762 0.0005405346494728943 0.0005402443021637564" /><geom name="C_6_R_collision" class="nonself_collision_primitive" type="sphere" size="0.064" />
          <body name="C_5" pos="0.013355 0 0.025525">
            <inertial pos="1.1666952453842032e-07 4.875120867284571e-10 -5.4527282094173864e-08" quat="0.6834639135686678 -0.18132037626643435 -0.18132037626643435 0.6834639135686678" mass="0.2517241492936801" diaginertia="0.0003309289611902095 0.00033092487739823986 0.00033061367874501045" /><geom name="C_5_R_collision" class="nonself_collision_primitive" type="sphere" size="0.058" />
            <body name="C_4" pos="0.010919 0 0.034482">
              <inertial pos="1.7996983387848192e-07 9.302906876956891e-10 1.420615170138086e-09" quat="0.7067175702458626 -0.023457960392690158 -0.023457960392690158 0.7067175702458626" mass="0.18541055898409603" diaginertia="0.00019283734172652068 0.00019248325263372215 0.0001923145499795622" /><geom name="C_4_R_collision" class="nonself_collision_primitive" type="sphere" size="0.052" />
              <body name="C_3" pos="0.0062566 0 0.036103">
                <inertial pos="7.145048379355792e-08 3.3800606587415965e-09 4.625297173804891e-07" quat="0.6750726211001228 -0.21042090257626656 -0.21042090257626656 0.6750726211001228" mass="0.14680851744209733" diaginertia="0.0001072992193368081 0.00010704563970869048 0.00010629536606613503" /><geom name="C_3_R_collision" class="nonself_collision_primitive" type="sphere" size="0.046" />
                <body name="C_2" pos="0.0075908 0 0.036994">
                  <inertial pos="3.713913884848836e-07 6.503774739634784e-09 -1.0952400400319942e-06" quat="0.6522043550176277 0.27318396603029554 0.27318396603029554 0.6522043550176277" mass="0.11663716116863002" diaginertia="6.380055604470059e-05 6.103956233638263e-05 5.7198167231206574e-05" /><geom name="C_2_R_collision" class="nonself_collision_primitive" type="sphere" size="0.04" />
                  <body name="C_1" pos="0.023095 0 0.032123">
                    <inertial pos="-1.7918285892767498e-06 3.371517653302703e-08 -9.526184496844653e-08" quat="0.9995579676595873 0.0 0.029729939257180447 0.0" mass="0.08861719688115782" diaginertia="3.697124819145177e-05 3.150839419677836e-05 3.0323080679393634e-05" /><geom name="C_1_R_collision" class="nonself_collision_primitive" type="sphere" size="0.034" />
                    <joint name="C_1_extend" class="cervical_extend" pos="-0.011471 0 -0.015955" axis="0 1 0" />
                    <joint name="C_1_bend" class="cervical_bend" pos="-0.011471 0 -0.015955" axis="-0.81194 0 0.58375" />
                    <body name="skull" pos="0.077928 0 0.017833">
                      <inertial pos="-0.0023917 0.0 0.0023167" quat="0.0 0.8164689904367176 0.0 0.5773892860585892" mass="0.3579225852102398" diaginertia="0.00072864 0.00064468 0.00028384" /><geom name="iris_L" class="visible_bone" type="ellipsoid" size="0.003 0.007 0.007" rgba="0.45 0.45 0.225 0.4" pos="0.023 0.027 0.01" euler="0 0 20" />
                      <geom name="pupil_L" class="visible_bone" type="sphere" size="0.003 0 0" rgba="0 0 0 1" pos="0.0215 0.0275 0.01" />
                      <geom name="iris_R" class="visible_bone" type="ellipsoid" size="0.003 0.007 0.007" rgba="0.45 0.45 0.225 0.4" pos="0.023 -0.027 0.01" euler="0 0 -20" />
                      <geom name="pupil_R" class="visible_bone" type="sphere" size="0.003 0 0" rgba="0 0 0 1" pos="0.0215 -0.0275 0.01" />
                      <geom name="skull0_collision" class="collision_primitive" type="ellipsoid" size="0.06 0.06 0.04" pos="-0.02 0 0.01" euler="0 10 0" />
                      <geom name="skull1_collision" class="collision_primitive" type="capsule" size="0.015 0.04 0.015" pos="0.06 0 -0.01" euler="0 110 0" />
                      <geom name="skull2_collision" class="collision_primitive" type="box" size="0.03 0.028 0.008" pos="0.02 0 -0.03" />
                      <geom name="skull3_collision" class="collision_primitive" type="box" size="0.02 0.018 0.006" pos="0.07 0 -0.03" />
                      <geom name="skull4_collision" class="collision_primitive" type="box" size="0.005 0.015 0.004" pos="0.095 0 -0.03" />
                      <joint name="atlas" class="atlas" pos="-0.062463 0 0" />
                      <site name="head" class="sensor" type="box" size="0.01 0.01 0.01" />
                      <site name="upper_bite" class="sensor" size="0.005" pos="0.065 0 -0.07" />
                      <body name="jaw" pos="0.0082923 0 -0.058237">
                        <inertial pos="-0.0015134 0.0 -0.00096033" quat="0.0 0.8970595103401998 0.0 0.441909758783624" mass="0.1332899051442993" diaginertia="0.0002144 0.00018551 6.3481e-05" /><geom name="jaw0_collision" class="collision_primitive" type="box" size="0.03 0.028 0.008" pos="-0.03 0 0.01" euler="0 55 0" />
                        <geom name="jaw1_collision" class="collision_primitive" type="box" size="0.02 0.022 0.005" pos="0 0 -0.012" euler="0 30 0" />
                        <geom name="jaw2_collision" class="collision_primitive" type="box" size="0.02 0.018 0.005" pos="0.03 0 -0.028" euler="0 25 0" />
                        <geom name="jaw3_collision" class="collision_primitive" type="box" size="0.015 0.013 0.003" pos="0.052 0 -0.035" euler="0 15 0" />
                        <joint name="mandible" class="mandible" pos="-0.043 0 0.05" axis="0 1 0" />
                        <site name="lower_bite" class="sensor" size="0.005" pos="0.063 0 0.005" />
                        <geom name="Canine_Bottom_L_collision" class="tooth_primitive" type="ellipsoid" size="0.0023284 0.002986 0.011309" pos="0.060438 0.012773 -0.030276" quat="0.60217 0.20042 0.3918 0.66612" />
                        <geom name="Canine_Bottom_R_collision" class="tooth_primitive" type="ellipsoid" size="0.0023284 0.002986 0.011309" pos="0.060438 -0.012773 -0.030276" quat="0.66612 0.3918 0.20042 0.60217" />
                        <geom name="Incisors_Bottom_L_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0012789 0.0016917 0.0075464" pos="0.069553 0.0020266 -0.033364" quat="-0.0027879 0.50693 0.0047406 0.86197" />
                        <geom name="Incisors_Bottom_L_2_collision" class="tooth_primitive" type="ellipsoid" size="0.001254 0.0017182 0.0073036" pos="0.068765 0.0056498 -0.032947" quat="-0.24629 0.49406 -0.046715 0.8325" />
                        <geom name="Incisors_Bottom_L_3_collision" class="tooth_primitive" type="ellipsoid" size="0.001485 0.0018936 0.0075715" pos="0.066779 0.0093056 -0.031601" quat="0.79249 0.07161 0.44127 0.41486" />
                        <geom name="Incisors_Bottom_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0012789 0.0016917 0.0075464" pos="0.069553 -0.0020265 -0.033364" quat="0.0027879 0.50693 -0.0047405 0.86197" />
                        <geom name="Incisors_Bottom_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.001254 0.0017182 0.0073036" pos="0.068765 -0.0056497 -0.032947" quat="0.24629 0.49406 0.046715 0.8325" />
                        <geom name="Incisors_Bottom_R_3_collision" class="tooth_primitive" type="ellipsoid" size="0.001485 0.0018936 0.0075715" pos="0.066779 -0.0093056 -0.031601" quat="0.41486 0.44127 0.07161 0.79249" />
                        <geom name="Molars_Bottom_3L_collision" class="tooth_primitive" type="ellipsoid" size="0.0018005 0.0026907 0.0027199" pos="-0.014575 0.020996 0.0073694" quat="0.56708 0.56708 0.4224 0.4224" />
                        <geom name="Molars_Bottom_L_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0028795 0.0087141 0.010424" pos="0.0027877 0.01936 -0.0074155" quat="0.42653 0.5875 0.52952 0.43877" />
                        <geom name="Molars_Bottom_L_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0026497 0.0039836 0.0048444" pos="-0.010552 0.020402 0.00095734" quat="0.70033 0.33849 0.27393 0.56562" />
                        <geom name="Molars_Bottom_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0028795 0.0087141 0.010424" pos="0.0027877 -0.01936 -0.0074155" quat="0.43877 0.52952 0.5875 0.42653" />
                        <geom name="Molars_Bottom_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0026497 0.0039836 0.0048444" pos="-0.010552 -0.020402 0.00095734" quat="0.56562 0.27393 0.33849 0.70033" />
                        <geom name="Molars_Bottom_R_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0018005 0.0026907 0.0027199" pos="-0.014575 -0.020996 0.0073694" quat="0.4224 0.4224 0.56708 0.56708" />
                        <geom name="Premolars_Bottom_L_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0012836 0.0020381 0.0043778" pos="0.04749 0.012799 -0.027576" quat="0.72376 0.083786 0.21501 0.65032" />
                        <geom name="Premolars_Bottom_L_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0020521 0.0051826 0.0057164" pos="0.039534 0.01408 -0.024161" quat="0.60291 0.47483 0.39424 0.50558" />
                        <geom name="Premolars_Bottom_L_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0020628 0.0052806 0.0059829" pos="0.028975 0.016126 -0.019807" quat="0.64746 0.40808 0.34017 0.54639" />
                        <geom name="Premolars_Bottom_L_4_collision" class="tooth_primitive" type="ellipsoid" size="0.002223 0.0060011 0.0069385" pos="0.017719 0.018186 -0.015" quat="0.62015 0.43869 0.37559 0.53095" />
                        <geom name="Premolars_Bottom_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0012836 0.0020381 0.0043778" pos="0.04749 -0.012799 -0.027576" quat="0.65032 0.21501 0.083786 0.72376" />
                        <geom name="Premolars_Bottom_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0020521 0.0051826 0.0057164" pos="0.039534 -0.014079 -0.024161" quat="0.50558 0.39424 0.47483 0.60291" />
                        <geom name="Premolars_Bottom_R_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0020628 0.0052806 0.0059829" pos="0.028975 -0.016126 -0.019807" quat="0.54639 0.34017 0.40808 0.64746" />
                        <geom name="Premolars_Bottom_R_4_collision" class="tooth_primitive" type="ellipsoid" size="0.002223 0.0060011 0.0069385" pos="0.017719 -0.018186 -0.015" quat="0.53095 0.37559 0.43869 0.62015" />
                        </body>
                      <geom name="Canine_Top_L_collision" class="tooth_primitive" type="ellipsoid" size="0.0025499 0.0043559 0.014793" pos="0.084079 0.014638 -0.037877" quat="0.72471 -0.012602 -0.14496 0.67351" />
                      <geom name="Canine_Top_R_collision" class="tooth_primitive" type="ellipsoid" size="0.0025499 0.0043559 0.014793" pos="0.084079 -0.014638 -0.037877" quat="0.67351 -0.14496 -0.012602 0.72471" />
                      <geom name="Incisors_Top_L_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0017755 0.0021365 0.0054398" pos="0.10162 0.0027916 -0.038244" quat="0.97791 0.023841 -0.19858 -0.060711" />
                      <geom name="Incisors_Top_L_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0020262 0.0022059 0.0057043" pos="0.10006 0.0077809 -0.038121" quat="0.97744 0.11358 -0.1769 0.020556" />
                      <geom name="Incisors_Top_L_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0023739 0.0027155 0.0070594" pos="0.095636 0.012224 -0.038721" quat="0.92895 0.17418 0.005002 -0.32664" />
                      <geom name="Incisors_Top_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0017755 0.0021365 0.0054398" pos="0.10162 -0.0027915 -0.038244" quat="0.97791 -0.023841 -0.19858 0.060711" />
                      <geom name="Incisors_Top_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0020262 0.0022059 0.0057043" pos="0.10006 -0.0077808 -0.038121" quat="0.97744 -0.11358 -0.1769 -0.020556" />
                      <geom name="Incisors_Top_R_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0023739 0.0027155 0.0070594" pos="0.095636 -0.012224 -0.038721" quat="0.92895 -0.17418 0.005002 0.32664" />
                      <geom name="Molars_Top_1L_collision" class="tooth_primitive" type="ellipsoid" size="0.0046165 0.0062275 0.0065298" pos="0.0197 0.027961 -0.040326" quat="-0.16961 0.79395 0.22795 0.53751" />
                      <geom name="Molars_Top_2L_collision" class="tooth_primitive" type="ellipsoid" size="0.003681 0.0041279 0.0046702" pos="0.010745 0.026296 -0.035745" quat="0.7973 0.5314 0.26968 0.095941" />
                      <geom name="Molars_Top_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0046165 0.0062275 0.0065298" pos="0.0197 -0.027961 -0.040326" quat="0.16961 0.79395 -0.22795 0.53751" />
                      <geom name="Molars_Top_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.003681 0.0041279 0.0046702" pos="0.010745 -0.026296 -0.035745" quat="0.5314 0.7973 -0.095942 -0.26968" />
                      <geom name="Premolars_Top_L_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0017159 0.0029358 0.0055561" pos="0.073229 0.015812 -0.039091" quat="0.72764 -0.013978 -0.13084 0.67322" />
                      <geom name="Premolars_Top_L_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0022037 0.0054716 0.0060794" pos="0.063298 0.017151 -0.039565" quat="0.40653 0.68555 0.52903 0.29136" />
                      <geom name="Premolars_Top_L_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0022548 0.0059774 0.0065176" pos="0.05089 0.021051 -0.03949" quat="0.34888 0.78683 0.47043 0.19462" />
                      <geom name="Premolars_Top_L_4_collision" class="tooth_primitive" type="ellipsoid" size="0.0036198 0.0076569 0.010082" pos="0.035357 0.025889 -0.040495" quat="0.54615 0.58019 0.43612 0.41821" />
                      <geom name="Premolars_Top_R_1_collision" class="tooth_primitive" type="ellipsoid" size="0.0017159 0.0029358 0.0055561" pos="0.073229 -0.015812 -0.039091" quat="0.67322 -0.13084 -0.013978 0.72764" />
                      <geom name="Premolars_Top_R_2_collision" class="tooth_primitive" type="ellipsoid" size="0.0022037 0.0054716 0.0060794" pos="0.063298 -0.017151 -0.039565" quat="0.29136 0.52903 0.68555 0.40653" />
                      <geom name="Premolars_Top_R_3_collision" class="tooth_primitive" type="ellipsoid" size="0.0022548 0.0059774 0.0065176" pos="0.05089 -0.021051 -0.03949" quat="0.19462 0.47043 0.78683 0.34888" />
                      <geom name="Premolars_Top_R_7_collision" class="tooth_primitive" type="ellipsoid" size="0.0036198 0.0076569 0.010082" pos="0.035357 -0.025889 -0.040495" quat="0.41821 0.43612 0.58019 0.54615" />
                      </body>
                  </body>
                  <joint name="C_2_twist" class="cervical_twist" pos="-0.0048286 0 -0.023533" axis="0.201 0 0.97959" />
                </body>
                <joint name="C_3_extend" class="cervical_extend" pos="-0.0028252 0 -0.016302" axis="0 1 0" />
                <joint name="C_3_bend" class="cervical_bend" pos="-0.0028252 0 -0.016302" axis="-0.98531 0 0.17075" />
              </body>
              <joint name="C_4_twist" class="cervical_twist" pos="-0.0050618 0 -0.015985" axis="0.30188 0 0.95334" />
            </body>
            <joint name="C_5_extend" class="cervical_extend" pos="-0.0074555 0 -0.01425" axis="0 1 0" />
            <joint name="C_5_bend" class="cervical_bend" pos="-0.0074555 0 -0.01425" axis="-0.88605 0 0.46359" />
          </body>
          <joint name="C_6_twist" class="cervical_twist" pos="-0.013896 0 -0.0071356" axis="0.88957 0 0.4568" />
        </body>
        <joint name="C_7_extend" class="cervical_extend" pos="-0.015711 0 -0.0016227" axis="0 1 0" />
        <joint name="C_7_bend" class="cervical_bend" pos="-0.015711 0 -0.0016227" axis="-0.10274 0 0.99471" />
      </body>
      <body name="scapula_L" pos="0.08 0.02 0.14">
        <inertial pos="0.03882287989522285 0.030832857552084304 -0.06492807185843928" quat="0.73468589906726 -0.16443904100392448 -0.2716916750600932 0.5994831650751198" mass="0.17658609814710902" diaginertia="0.0002589663969446711 0.0002148022615416103 6.794365455559313e-05" /><joint name="scapula_L_supinate" class="scapula_supinate" pos="-0.03 -0.02 0" axis="0 0 1" />
        <joint name="scapula_L_abduct" class="scapula_abduct" pos="0 -0.05 -0.07" axis="1 0 0" />
        <joint name="scapula_L_extend" class="scapula_extend" pos="0 0 0" axis="0.3 1 0" />
        <body name="upper_arm_L" pos="0.075 0.033 -0.13">
          <inertial pos="-0.018022709535955406 0.009850001688954061 -0.0770378578339447" quat="0.5691802371268407 0.17822067310264925 0.08000399179072756 0.7986680227983479" mass="0.1757417186851815" diaginertia="0.0005507450953562994 0.0005473006139616115 2.4222196538866016e-05" /><joint name="shoulder_L_supinate" class="shoulder_supinate" axis="-1 0 1" />
          <joint name="shoulder_L_extend" class="shoulder_extend" axis="0 1 0" />
          <body name="lower_arm_L" pos="-0.05 0.015 -0.145">
            <inertial pos="-0.0043986884926116646 -0.0043785234213789305 -0.08359474952312357" quat="0.7355522793933432 -0.05248580054666463 0.0072501943723538124 0.6753928632298368" mass="0.16547619481522657" diaginertia="0.0006356732422655886 0.0006314149802599816 2.1214056021447423e-05" /><joint name="elbow_L" class="elbow" axis="0 1 0.2" />
            <body name="hand_anchor_L" pos="0.003 -0.015 -0.19">
              <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.00030000000000000003" diaginertia="5.000000000000001e-09 5.000000000000001e-09 5.000000000000001e-09" /><geom name="hand_anchor_L" class="foot_primitive" type="box" contype="0" conaffinity="0" size="0.005 0.005 0.005" />
              <site name="hand_anchor_L" class="sensor" />
              <body name="hand_L">
                <inertial pos="0.009877055671260487 0.0003319364495930286 -0.034841558231151655" quat="0.9783555915072186 0.003633215845912681 -0.1896326720844105 0.0827440994107963" mass="0.02944092481504093" diaginertia="1.467922865435897e-05 1.3048678459610214e-05 3.5812422370820533e-06" /><joint name="wrist_L" class="wrist" axis="0 -1 0" />
                <geom name="hand_L_collision" class="collision_primitive" type="box" size="0.03 0.016 0.012" pos="0.01 0 -0.04" euler="0 65 0" />
                <body name="finger_L" pos="0.02 0 -0.06">
                  <inertial pos="0.02148002499704705 0.0012592563844674221 -0.013638015870860594" quat="-0.030208163262606298 0.8161034746316684 0.014631963652976634 0.5769302307957865" mass="0.028553834898591646" diaginertia="1.1337122104505103e-05 7.449901512505357e-06 7.159506802428641e-06" /><joint name="finger_L" class="finger" axis="0 1 0.2" />
                  <geom name="finger_L0_collision" class="foot_primitive" size="0.018 0.012" pos="0.012 0 -0.012" euler="90 0 0" />
                  <geom name="finger_L1_collision" class="foot_primitive" size="0.01 0.015" pos="0.032 0 -0.02" euler="90 0 0" />
                  <geom name="finger_L2_collision" class="foot_primitive" size="0.008 0.01" pos="0.042 0 -0.022" euler="90 0 0" />
                  <site name="palm_L" class="sensor" type="box" size="0.028 0.03 0.007" pos="0.02 0 -0.024" />
                </body>
              </body>
            </body>
            <geom name="Radius_L_collision" class="collision_primitive" size="0.017 0.09" pos="-0.0016515 -0.0069684 -0.10807" quat="0.9802 -0.056819 0.010884 0.18935" />
            <geom name="Ulna_L_collision" class="collision_primitive" size="0.015 0.06" pos="-0.01187 -0.00044152 -0.051646" quat="0.69933 -0.032243 -0.022055 0.71373" />
          </body>
          <geom name="humerus_L_collision" class="collision_primitive" size="0.02 0.08" pos="-0.018023 0.00985 -0.077038" quat="0.56918 0.17822 0.080004 0.79867" />
        </body>
        <geom name="Scapula_L_collision" class="collision_primitive" type="box" size="0.01758 0.03491 0.057492" pos="0.038823 0.030833 -0.064928" quat="0.73469 -0.16444 -0.27169 0.59948" />
      </body>
      <body name="scapula_R" pos="0.08 -0.02 0.14">
        <inertial pos="0.03882288185419291 -0.030832843303741742 -0.06492807256864762" quat="0.5994831354398243 -0.2716916496491441 -0.1644390340050473 0.7346859342124717" mass="0.17658609328336178" diaginertia="0.00025896637822749854 0.00021480224750196081 6.79436497234081e-05" /><joint name="scapula_R_supinate" class="scapula_supinate" pos="-0.03 0.02 0" axis="0 0 -1" />
        <joint name="scapula_R_abduct" class="scapula_abduct" pos="0 0.05 -0.07" axis="-1 0 0" />
        <joint name="scapula_R_extend" class="scapula_extend" pos="0 0 0" axis="-0.3 1 0" />
        <body name="upper_arm_R" pos="0.075 -0.033 -0.13">
          <inertial pos="-0.01802270734152448 -0.009849982881453948 -0.07703785366332667" quat="0.7986679664545769 0.080003997101158 0.17822067155398094 0.5691803159263312" mass="0.17574172836518198" diaginertia="0.0005507451159286501 0.0005473006346846905 2.4222197890646892e-05" /><joint name="shoulder_R_supinate" class="shoulder_supinate" axis="1 0 -1" />
          <joint name="shoulder_R_extend" class="shoulder_extend" axis="0 1 0" />
          <body name="lower_arm_R" pos="-0.05 -0.015 -0.145">
            <inertial pos="-0.004398687254573189 0.004378542206248238 -0.08359475158691031" quat="0.6753929636091572 0.007250190562324147 -0.05248577837597138 0.7355521888434017" mass="0.1654761604905651" diaginertia="0.000635673077340521 0.0006314148157958271 2.1214048940094887e-05" /><joint name="elbow_R" class="elbow" axis="0 1 -0.2" />
            <body name="hand_anchor_R" pos="0.003 0.015 -0.19">
              <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.00030000000000000003" diaginertia="5.000000000000001e-09 5.000000000000001e-09 5.000000000000001e-09" /><geom name="hand_anchor_R" class="foot_primitive" type="box" contype="0" conaffinity="0" size="0.005 0.005 0.005" />
              <site name="hand_anchor_R" class="sensor" />
              <body name="hand_R">
                <inertial pos="0.009877056106461123 -0.00033192086653820814 -0.03484155894194094" quat="0.9783555707111999 -0.0036332638154777507 -0.18963267332522085 -0.08274434034989818" mass="0.02944092399464633" diaginertia="1.467922792582732e-05 1.3048677783609827e-05 3.5812416735707634e-06" /><joint name="wrist_R" class="wrist" axis="0 -1 0" />
                <geom name="hand_R_collision" class="collision_primitive" type="box" size="0.03 0.016 0.012" pos="0.01 0 -0.04" euler="0 65 0" />
                <body name="finger_R" pos="0.02 0 -0.06">
                  <inertial pos="0.02148002509919743 -0.0012592472906654893 -0.013638015732182663" quat="0.03020534744911452 0.8161035760666555 -0.014630148380879233 0.5769302807749411" mass="0.028553835365184285" diaginertia="1.133712084670667e-05 7.449901331386501e-06 7.1595058680466126e-06" /><joint name="finger_R" class="finger" axis="0 1 -0.2" />
                  <geom name="finger_R0_collision" class="foot_primitive" size="0.018 0.012" pos="0.012 0 -0.012" euler="90 0 0" />
                  <geom name="finger_R1_collision" class="foot_primitive" size="0.01 0.015" pos="0.032 0 -0.02" euler="90 0 0" />
                  <geom name="finger_R2_collision" class="foot_primitive" size="0.008 0.01" pos="0.042 0 -0.022" euler="90 0 0" />
                  <site name="palm_R" class="sensor" type="box" size="0.028 0.03 0.007" pos="0.02 0 -0.024" />
                </body>
              </body>
            </body>
            <geom name="Radius_R_collision" class="collision_primitive" size="0.017 0.09" pos="-0.0016515 0.0069684 -0.10807" quat="0.9802 0.056819 0.010884 -0.18935" />
            <geom name="Ulna_R_collision" class="collision_primitive" size="0.015 0.06" pos="-0.01187 0.00044155 -0.051646" quat="0.71373 -0.022055 -0.032243 0.69933" />
          </body>
          <geom name="humerus_R_collision" class="collision_primitive" size="0.02 0.08" pos="-0.018023 -0.00985 -0.077038" quat="0.79867 0.080004 0.17822 0.56918" />
        </body>
        <geom name="Scapula_R_collision" class="collision_primitive" type="box" size="0.01758 0.03491 0.057492" pos="0.038823 -0.030833 -0.064928" quat="0.59948 -0.27169 -0.16444 0.73469" />
      </body>
      </body>
    <body name="ball" pos="0.5 0 0.1">
      <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.056" diaginertia="2.016e-05 2.016e-05 2.016e-05" /><freejoint name="ball_root" />
      <geom name="ball" class="bouncy" size="0.03" material="tennis_ball" mass="0.056" />
    </body>
    <geom name="target" class="velcro" type="cylinder" size="0.1 0.004" material="target" />
    <geom name="wall_px" type="plane" size="1 10 0.5" material="decoration" pos="-10.7 0 0.7" zaxis="1 0 1" />
    <geom name="wall_py" type="plane" size="10 1 0.5" material="decoration" pos="0 -10.7 0.7" zaxis="0 1 1" />
    <geom name="wall_nx" type="plane" size="1 10 0.5" material="decoration" pos="10.7 0 0.7" zaxis="-1 0 1" />
    <geom name="wall_ny" type="plane" size="10 1 0.5" material="decoration" pos="0 10.7 0.7" zaxis="0 -1 1" />
    <camera name="ball" mode="targetbodycom" target="ball" pos="0.5 0.5 0.9" />
    <camera name="head" mode="targetbodycom" target="skull" pos="-0.5 -0.5 0.9" />
  </worldbody>
  <contact>
    <exclude name="upper_arm_L:torso" body1="upper_arm_L" body2="torso" />
    <exclude name="upper_arm_R:torso" body1="upper_arm_R" body2="torso" />
    <exclude name="pelvis:Ca_2" body1="pelvis" body2="Ca_2" />
    <exclude name="pelvis:Ca_3" body1="pelvis" body2="Ca_3" />
    <exclude name="Ca_1:Ca_3" body1="Ca_1" body2="Ca_3" />
    <exclude name="Ca_2:Ca_4" body1="Ca_2" body2="Ca_4" />
    <exclude name="Ca_3:Ca_5" body1="Ca_3" body2="Ca_5" />
    <exclude name="Ca_4:Ca_6" body1="Ca_4" body2="Ca_6" />
    <exclude name="C_7:scapula_L" body1="C_7" body2="scapula_L" />
    <exclude name="C_7:scapula_R" body1="C_7" body2="scapula_R" />
    <exclude name="C_6:scapula_L" body1="C_6" body2="scapula_L" />
    <exclude name="C_6:scapula_R" body1="C_6" body2="scapula_R" />
    <exclude name="C_5:scapula_L" body1="C_5" body2="scapula_L" />
    <exclude name="C_5:scapula_R" body1="C_5" body2="scapula_R" />
    <exclude name="C_1:jaw" body1="C_1" body2="jaw" />
    <exclude name="torso:lower_arm_L" body1="torso" body2="lower_arm_L" />
    <exclude name="torso:lower_arm_R" body1="torso" body2="lower_arm_R" />
    <exclude name="C_4:scapula_R" body1="C_4" body2="scapula_R" />
    <exclude name="C_4:scapula_L" body1="C_4" body2="scapula_L" />
    <exclude name="C_5:upper_arm_R" body1="C_5" body2="upper_arm_R" />
    <exclude name="C_5:upper_arm_L" body1="C_5" body2="upper_arm_L" />
    <exclude name="C_6:upper_arm_R" body1="C_6" body2="upper_arm_R" />
    <exclude name="C_6:upper_arm_L" body1="C_6" body2="upper_arm_L" />
    <exclude name="C_7:upper_arm_R" body1="C_7" body2="upper_arm_R" />
    <exclude name="C_7:upper_arm_L" body1="C_7" body2="upper_arm_L" />
    <exclude name="upper_leg_L:upper_leg_R" body1="upper_leg_L" body2="upper_leg_R" />
    <exclude name="lower_leg_L:pelvis" body1="lower_leg_L" body2="pelvis" />
    <exclude name="upper_leg_L:foot_L" body1="upper_leg_L" body2="foot_L" />
    <exclude name="lower_leg_R:pelvis" body1="lower_leg_R" body2="pelvis" />
    <exclude name="upper_leg_R:foot_R" body1="upper_leg_R" body2="foot_R" />
  </contact>
  <tendon>
    <fixed name="lumbar_extend" class="lumbar_extend">
      <joint joint="L_1_extend" coef="1.0887906214976453" />
      <joint joint="L_3_extend" coef="1.0287221260745385" />
      <joint joint="L_5_extend" coef="0.9678084616051313" />
      <joint joint="L_7_extend" coef="0.9146787908226849" />
    </fixed>
    <fixed name="lumbar_bend" class="lumbar_bend">
      <joint joint="L_1_bend" coef="1.1360135211286146" />
      <joint joint="L_3_bend" coef="1.0490053091600777" />
      <joint joint="L_5_bend" coef="0.9555458104917836" />
      <joint joint="L_7_bend" coef="0.8594353592195243" />
    </fixed>
    <fixed name="lumbar_twist" class="lumbar_twist">
      <joint joint="L_2_twist" coef="1.0007967860214304" />
      <joint joint="L_4_twist" coef="0.9996855391691534" />
      <joint joint="L_6_twist" coef="0.9995176748094166" />
    </fixed>
    <fixed name="cervical_extend" class="cervical_extend">
      <joint joint="C_7_extend" coef="1.1651729548786316" />
      <joint joint="C_5_extend" coef="1.0638262565660335" />
      <joint joint="C_3_extend" coef="0.9279175112069821" />
      <joint joint="C_1_extend" coef="0.8430832773483526" />
    </fixed>
    <fixed name="cervical_bend" class="cervical_bend">
      <joint joint="C_7_bend" coef="1.1032094508820511" />
      <joint joint="C_5_bend" coef="1.1151582521827508" />
      <joint joint="C_3_bend" coef="0.9257028175303736" />
      <joint joint="C_1_bend" coef="0.8559294794048242" />
    </fixed>
    <fixed name="cervical_twist" class="cervical_twist">
      <joint joint="C_6_twist" coef="1.0490139787418138" />
      <joint joint="C_4_twist" coef="0.9679313738406048" />
      <joint joint="C_2_twist" coef="0.9830546474175812" />
    </fixed>
    <fixed name="caudal_extend" class="caudal_extend">
      <joint joint="Ca_1_extend" coef="1.4175697632800168" />
      <joint joint="Ca_3_extend" coef="1.3533291916469086" />
      <joint joint="Ca_5_extend" coef="1.2905602895468757" />
      <joint joint="Ca_7_extend" coef="1.1963357977448668" />
      <joint joint="Ca_9_extend" coef="1.0882282181745322" />
      <joint joint="Ca_11_extend" coef="0.9549416058705948" />
      <joint joint="Ca_13_extend" coef="0.839068010247595" />
      <joint joint="Ca_15_extend" coef="0.7575627896787993" />
      <joint joint="Ca_17_extend" coef="0.71516710400689" />
      <joint joint="Ca_19_extend" coef="0.6955591082173007" />
      <joint joint="Ca_21_extend" coef="0.6916781215856203" />
    </fixed>
    <fixed name="caudal_bend" class="caudal_bend">
      <joint joint="Ca_2_bend" coef="1.3236846643142188" />
      <joint joint="Ca_4_bend" coef="1.3130782401287662" />
      <joint joint="Ca_6_bend" coef="1.2679178338493764" />
      <joint joint="Ca_8_bend" coef="1.170681278111726" />
      <joint joint="Ca_10_bend" coef="1.0404777760545052" />
      <joint joint="Ca_12_bend" coef="0.9053320718072471" />
      <joint joint="Ca_14_bend" coef="0.8040064645300866" />
      <joint joint="Ca_16_bend" coef="0.7475697729484704" />
      <joint joint="Ca_18_bend" coef="0.7188107117102569" />
      <joint joint="Ca_20_bend" coef="0.7084411865453454" />
    </fixed>
  </tendon>
  <actuator>
    <general name="lumbar_extend" class="lumbar_extend" tendon="lumbar_extend" />
    <general name="lumbar_bend" class="lumbar_bend" tendon="lumbar_bend" />
    <general name="lumbar_twist" class="lumbar_twist" tendon="lumbar_twist" />
    <general name="cervical_extend" class="cervical_extend" tendon="cervical_extend" />
    <general name="cervical_bend" class="cervical_bend" tendon="cervical_bend" />
    <general name="cervical_twist" class="cervical_twist" tendon="cervical_twist" />
    <general name="caudal_extend" class="caudal_extend" tendon="caudal_extend" />
    <general name="caudal_bend" class="caudal_bend" tendon="caudal_bend" />
    <general name="hip_L_supinate" class="hip_supinate" joint="hip_L_supinate" />
    <general name="hip_L_abduct" class="hip_abduct" joint="hip_L_abduct" />
    <general name="hip_L_extend" class="hip_extend" joint="hip_L_extend" />
    <general name="knee_L" class="knee" joint="knee_L" />
    <general name="ankle_L" class="ankle" joint="ankle_L" />
    <general name="toe_L" class="toe" joint="toe_L" />
    <general name="hip_R_supinate" class="hip_supinate" joint="hip_R_supinate" />
    <general name="hip_R_abduct" class="hip_abduct" joint="hip_R_abduct" />
    <general name="hip_R_extend" class="hip_extend" joint="hip_R_extend" />
    <general name="knee_R" class="knee" joint="knee_R" />
    <general name="ankle_R" class="ankle" joint="ankle_R" />
    <general name="toe_R" class="toe" joint="toe_R" />
    <general name="atlas" class="atlas" joint="atlas" />
    <general name="mandible" class="mandible" joint="mandible" />
    <general name="scapula_L_supinate" class="scapula_supinate" joint="scapula_L_supinate" />
    <general name="scapula_L_abduct" class="scapula_abduct" joint="scapula_L_abduct" />
    <general name="scapula_L_extend" class="scapula_extend" joint="scapula_L_extend" />
    <general name="shoulder_L_supinate" class="shoulder_supinate" joint="shoulder_L_supinate" />
    <general name="shoulder_L_extend" class="shoulder_extend" joint="shoulder_L_extend" />
    <general name="elbow_L" class="elbow" joint="elbow_L" />
    <general name="wrist_L" class="wrist" joint="wrist_L" />
    <general name="finger_L" class="finger" joint="finger_L" />
    <general name="scapula_R_supinate" class="scapula_supinate" joint="scapula_R_supinate" />
    <general name="scapula_R_abduct" class="scapula_abduct" joint="scapula_R_abduct" />
    <general name="scapula_R_extend" class="scapula_extend" joint="scapula_R_extend" />
    <general name="shoulder_R_supinate" class="shoulder_supinate" joint="shoulder_R_supinate" />
    <general name="shoulder_R_extend" class="shoulder_extend" joint="shoulder_R_extend" />
    <general name="elbow_R" class="elbow" joint="elbow_R" />
    <general name="wrist_R" class="wrist" joint="wrist_R" />
    <general name="finger_R" class="finger" joint="finger_R" />
  </actuator>
  <sensor>
    <accelerometer name="accelerometer" site="head" />
    <velocimeter name="velocimeter" site="head" />
    <gyro name="gyro" site="head" />
    <subtreelinvel name="torso_linvel" body="torso" />
    <subtreeangmom name="torso_angmom" body="torso" />
    <touch name="palm_L" site="palm_L" />
    <touch name="palm_R" site="palm_R" />
    <touch name="sole_L" site="sole_L" />
    <touch name="sole_R" site="sole_R" />
    <force name="foot_L" site="foot_anchor_L" />
    <force name="foot_R" site="foot_anchor_R" />
    <force name="hand_L" site="hand_anchor_L" />
    <force name="hand_R" site="hand_anchor_R" />
  </sensor>
</mujoco>"""


comptime dm_dog_fetch_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _DOG_SKIN_ASSETS,
    _DOG_FETCH_ASSETS, _DOG_FETCH_BODY,
)

comptime dfp = parse_xml(dm_dog_fetch_xml)


# --- observation layout ------------------------------------------------------
#
# `Fetch.get_observation_components` is Stand's plus two entries:
#
#   (Stand)             223   see dog_xml.mojo's layout note
#   ball_state            6   ball_in_head_frame: position THEN velocity,
#                             both rotated into the head site frame
#   target_position       3   target_in_head_frame
#                       -----
#                         232
comptime DOG_FETCH_OBS_DIM: Int = 232


comptime DMDogFetchModel = ModelDefFromXML[
    xml=dm_dog_fetch_xml,
    nbody=dfp.NBODY, njoint=dfp.NJOINT, nq=dfp.NQ, nv=dfp.NV,
    ngeom=dfp.NGEOM, nact=dfp.NACT, ntex=dfp.NTEX, nmat=dfp.NMAT,
    nlight=dfp.NLIGHT, ncam=dfp.NCAM, nsite=dfp.NSITE,
    max_tendon=dfp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # dog's own four feet plus the ball against the floor, the walls, and
    # whatever limb is nudging it.
    max_contacts=28,
    obs_dim_override=DOG_FETCH_OBS_DIM,
    obs_qpos_skip=0,
    neq=dfp.NEQ,
    nexclude=dfp.NEXCLUDE,
    timestep=dfp.TIMESTEP,
    # ⚠ DERIVED, NEVER HAND-WRITTEN. The ball is `class="bouncy"` and the
    # target `class="velcro"`, both condim=6; without this they are silently
    # downgraded to four pyramid edges and the ball spins and rolls unopposed
    # (defect 004fe439, and defect 8 in a second dress). dog's own geoms
    # already carry condim 6 on 42 of 128, so this is not new to fetch — it is
    # simply not optional.
    max_condim=dfp.MAX_CONDIM,
]


# --- indices, read out of a COMPILED mjModel, never counted by hand ----------
#
# The ball is appended last, so dog's own qpos/qvel layout is untouched and
# only the tail is new — the same property quadruped fetch relies on.
comptime FETCH_BALL_BODY_IDX: Int = 62
comptime FETCH_BALL_QPOS_0: Int = 80
comptime FETCH_BALL_DOF_0: Int = 79

# `target` is declared in the worldbody BEFORE the dog, so it takes geom id 1
# (the floor is 0) and the ball, added last, is 133.
comptime FETCH_GEOM_BALL: Int = 133
comptime FETCH_GEOM_TARGET: Int = 1
comptime FETCH_GEOM_FLOOR: Int = 0

# Sites are UNCHANGED from stand — fetch adds none, so `head`, `upper_bite`
# and `lower_bite` keep their ids and dog_config's constants stay valid.
comptime FETCH_SITE_HEAD: Int = 5
comptime FETCH_SITE_UPPER_BITE: Int = 6
comptime FETCH_SITE_LOWER_BITE: Int = 7

# --- reward geometry, read from the compiled model --------------------------
# `bite_radius`  = site_size['upper_bite', 0]
# `target_radius`= geom_size['target', 0]
# `bring_margin` = geom_size['floor', 0]  (the floor HALF-extent, i.e. 10)
comptime FETCH_BITE_RADIUS: Float64 = 0.005
comptime FETCH_TARGET_RADIUS: Float64 = 0.1
comptime FETCH_BRING_MARGIN: Float64 = 10.0

# `Fetch.initialize_episode` throws the ball from 0.75 * floor half-extent.
comptime FETCH_THROW_RADIUS: Float64 = 0.75 * FETCH_BRING_MARGIN
comptime FETCH_THROW_HEIGHT_MAX: Float64 = 3.0
comptime FETCH_THROW_SPEED_MAX: Float64 = 5.0
comptime FETCH_BALL_SPAWN_Z: Float64 = 0.05

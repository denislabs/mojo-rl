"""MJCF XML parser for compile-time model dimension extraction and full parsing."""

from .xml_parser import ParsedModel, parse_xml, merge_mjcf, ComptimeRenderData, parse_xml_render_data
from .render_fields import RenderFields, build_render_fields
from .flat_model import (
    BodyData,
    JointData,
    GeomData,
    ActuatorData,
    TextureData,
    MaterialData,
    LightData,
    CameraData,
    SiteData,
    DefaultsData,
    EqualityData,
    NamedDefaultsList,
    FlatModelDef,
    TEX_SKYBOX,
    TEX_2D,
    TEX_CUBE,
    TEX_BUILTIN_NONE,
    TEX_BUILTIN_GRADIENT,
    TEX_BUILTIN_CHECKER,
    TEX_BUILTIN_FLAT,
    TEX_MARK_NONE,
    TEX_MARK_EDGE,
    TEX_MARK_CROSS,
    TEX_MARK_RANDOM,
    LIGHT_MODE_FIXED,
    LIGHT_MODE_TRACK,
    LIGHT_MODE_TRACKCOM,
    LIGHT_MODE_TARGETBODY,
    LIGHT_MODE_TARGETBODYCOM,
    CAM_MODE_FIXED,
    CAM_MODE_TRACK,
    CAM_MODE_TRACKCOM,
    CAM_MODE_TARGETBODY,
    CAM_MODE_TARGETBODYCOM,
)
from .full_parser import parse_xml_full
from .model_def_from_xml import ModelDefFromXML

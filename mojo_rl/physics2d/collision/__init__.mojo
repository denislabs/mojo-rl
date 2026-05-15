"""Collision detection implementations for physics simulation."""

from .flat_terrain import FlatTerrainCollision
from .edge_terrain import EdgeTerrainCollision, MAX_TERRAIN_EDGES
from .circle_polygon import (
    CirclePolygonCollision,
    detect_circle_vs_body_pair,
)

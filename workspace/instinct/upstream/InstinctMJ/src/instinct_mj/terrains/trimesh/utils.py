from __future__ import annotations

import trimesh


def crop_terrain_mesh_aabb(
    terrain_mesh: trimesh.Trimesh,
    x_max: float | None = None,
    x_min: float | None = None,
    y_max: float | None = None,
    y_min: float | None = None,
    z_max: float | None = None,
    z_min: float | None = None,
) -> trimesh.Trimesh:
    """Crop the terrain mesh to given the bounding box coordinates.
    Args:
        terrain_mesh (trimesh.Trimesh): The terrain mesh to be cropped.
        size (tuple[float, float]): The size of the bounding box (width, depth).
    Returns:
        trimesh.Trimesh: The cropped terrain mesh.
    """

    if x_max is not None:
        slice_plane_normal = [-1, 0, 0]
        slice_plane_origin = [x_max, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if x_min is not None:
        slice_plane_normal = [1, 0, 0]
        slice_plane_origin = [x_min, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_max is not None:
        slice_plane_normal = [0, -1, 0]
        slice_plane_origin = [0, y_max, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_min is not None:
        slice_plane_normal = [0, 1, 0]
        slice_plane_origin = [0, y_min, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_max is not None:
        slice_plane_normal = [0, 0, -1]
        slice_plane_origin = [0, 0, z_max]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_min is not None:
        slice_plane_normal = [0, 0, 1]
        slice_plane_origin = [0, 0, z_min]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    return terrain_mesh

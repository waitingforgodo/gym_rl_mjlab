from __future__ import annotations

from dataclasses import dataclass

from mjlab.terrains import TerrainEntity
from mjlab.terrains import TerrainEntityCfg as TerrainEntityCfgBase


@dataclass(kw_only=True)
class TerrainEntityCfg(TerrainEntityCfgBase):
    """mjlab terrain entity config carrying the ``class_type`` dispatch hook.

  IsaacLab's ``InteractiveScene`` builds every terrain through
  ``cfg.class_type(cfg)``, so the plain (non-importer) terrains in the source
  were overridable too. mjlab's own ``TerrainEntityCfg`` has no such field, so
  this subclass restores the hook. The default is mjlab's ``TerrainEntity``,
  which is what the scene constructed before.
  """

    class_type: type = TerrainEntity
    """The class to use for the terrain entity."""

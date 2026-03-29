"""
AniForm File Readers

Binary readers for AniForm simulation files:
- .afm  : Mesh file (reference configuration) — ReadAFMesh
- .afr  : Result field file (per-field binary) — ReadAFResult
- .afs  : Solution file (increment timing, convergence) — ReadAFSFile
- .msh  : Per-ply deformed mesh — find_ply_mesh_files (size only; class is heuristic)

ReadAFProject is NOT included — that was a speculative stub. Do not use .afp files.
See ANIFORM_REFERENCE.md §2 for full API documentation.
"""

from .ReadAFMesh   import ReadAFMesh
from .ReadAFResult import ReadAFResult
from .ReadAFSFile  import ReadAFSFile
from .ReadMSHFile  import find_ply_mesh_files

__all__ = [
    'ReadAFMesh',
    'ReadAFResult',
    'ReadAFSFile',
    'find_ply_mesh_files',
]

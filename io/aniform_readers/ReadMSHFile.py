"""
Reader for .msh Mesh Files (CGNS or Custom Format)

AniForm exports mesh files in .msh format which may be:
- Custom binary format
- Text-based format
- CGNS format

This reader attempts to parse these files and extract:
- Node coordinates
- Element connectivity
- Boundary conditions
- Material properties
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Any
import struct


class ReadMSHFile:
    """
    Reader for .msh mesh files.
    
    Handles multiple possible formats and extracts:
    - Node coordinates (x, y, z)
    - Element connectivity
    - Element groups/sections
    """
    
    def __init__(self):
        self.filepath: Optional[Path] = None
    
    def read(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a .msh file.
        
        Args:
            filepath: Path to the .msh file
            
        Returns:
            Dictionary containing:
            - 'nodes': (N, 3) array of node coordinates
            - 'elements': (M, connectivity_size) array
            - 'format': Detected format type
            - Additional format-specific data
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.filepath}")
        
        assert self.filepath is not None  # Type checker hint
        
        # Read first few bytes to determine format
        with open(self.filepath, 'rb') as f:
            header = f.read(100)
        
        # Check for text-based format
        try:
            header_text = header.decode('ascii')
            if '$' in header_text or 'MESH' in header_text.upper():
                return self._read_text_format()
        except:
            pass
        
        # Try binary format
        return self._read_binary_format()
    
    def _read_text_format(self) -> Dict[str, Any]:
        """Read text-based .msh file (GID or Gmsh-like format)."""
        mesh_data = {
            'nodes': None,
            'elements': None,
            'format': 'text',
        }
        
        assert self.filepath is not None
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        nodes_dict = {}  # Use dict to handle non-sequential node IDs
        elements_list = []
        
        reading_nodes = False
        reading_elements = False
        
        for line in lines:
            line = line.strip()
            
            # Check for section markers
            if 'coordinates' in line.lower():
                reading_nodes = True
                reading_elements = False
                continue
            elif 'elements' in line.lower():
                reading_nodes = False
                reading_elements = True
                continue
            elif line.startswith('end') or line.startswith('End') or line.startswith('END'):
                reading_nodes = False
                reading_elements = False
                continue
            
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Parse nodes: node_id x y z
            if reading_nodes:
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        nodes_dict[node_id] = [x, y, z]
                except (ValueError, IndexError):
                    continue
            
            # Parse elements: element_id node1 node2 node3
            elif reading_elements:
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        # Skip element ID, get node IDs
                        n1 = int(parts[1])
                        n2 = int(parts[2])
                        n3 = int(parts[3])
                        elements_list.append([n1, n2, n3])
                except (ValueError, IndexError):
                    continue
        
        # Convert nodes dict to array
        if nodes_dict:
            # Create mapping from original node IDs to sequential indices
            sorted_node_ids = sorted(nodes_dict.keys())
            node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
            
            # Build node array
            nodes_array = np.array([nodes_dict[nid] for nid in sorted_node_ids], dtype=np.float64)
            mesh_data['nodes'] = nodes_array
            mesh_data['node_id_mapping'] = node_id_to_idx
            mesh_data['original_node_ids'] = sorted_node_ids
            
            # Convert elements, remapping node IDs to sequential indices
            if elements_list:
                elements_remapped = []
                for elem in elements_list:
                    try:
                        remapped = [node_id_to_idx[nid] for nid in elem]
                        elements_remapped.append(remapped)
                    except KeyError:
                        # Skip elements referencing non-existent nodes
                        continue
                
                if elements_remapped:
                    mesh_data['elements'] = np.array(elements_remapped, dtype=np.int32)
        
        return mesh_data
    
    def _read_binary_format(self) -> Dict[str, Any]:
        """Read binary .msh file."""
        mesh_data = {
            'nodes': None,
            'elements': None,
            'format': 'binary',
        }
        
        assert self.filepath is not None
        with open(self.filepath, 'rb') as f:
            # Try to read header
            try:
                # Attempt different binary formats
                # Format 1: Standard binary with header
                version = struct.unpack('<I', f.read(4))[0]
                num_nodes = struct.unpack('<I', f.read(4))[0]
                num_elements = struct.unpack('<I', f.read(4))[0]
                
                if num_nodes > 0 and num_nodes < 10_000_000:  # Sanity check
                    # Read nodes
                    nodes_data = f.read(num_nodes * 3 * 8)
                    if len(nodes_data) == num_nodes * 3 * 8:
                        nodes = np.frombuffer(nodes_data, dtype='<f8').reshape(num_nodes, 3)
                        mesh_data['nodes'] = nodes.copy()
                    
                    # Read elements (assuming triangles)
                    elements_data = f.read(num_elements * 3 * 4)
                    if len(elements_data) == num_elements * 3 * 4:
                        elements = np.frombuffer(elements_data, dtype='<i4').reshape(num_elements, 3)
                        mesh_data['elements'] = elements.copy()
                
            except Exception as e:
                # Try alternative parsing
                f.seek(0)
                
                # Read entire file and try to parse
                data = f.read()
                
                # Look for patterns that might indicate nodes/elements
                # This is a heuristic approach
                mesh_data['raw_size'] = len(data)
                mesh_data['parse_error'] = str(e)
        
        return mesh_data


def read_msh_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to read a .msh file.
    
    Args:
        filepath: Path to the .msh file
        
    Returns:
        Mesh data dictionary
    """
    reader = ReadMSHFile()
    return reader.read(filepath)


def find_ply_mesh_files(timestep_dir: Path) -> Dict[int, Path]:
    """
    Find mesh files for each ply in a timestep directory.
    
    Looks for patterns like:
    - Part.section 1.1.msh (ply 1)
    - Part.section 1.2.msh (ply 2)
    - Part.section 1.3.msh (ply 3)
    
    Args:
        timestep_dir: Path to timestep directory
        
    Returns:
        Dictionary mapping ply index (0, 1, 2) to mesh file path
    """
    meshes_dir = timestep_dir / "meshes"
    
    if not meshes_dir.exists():
        return {}
    
    ply_meshes = {}
    
    # Look for Part.section X.Y.msh files
    for msh_file in meshes_dir.glob("Part.section *.*.msh"):
        # Extract ply number from filename
        # Format: Part.section 1.1.msh, Part.section 1.2.msh, etc.
        parts = msh_file.stem.split('.')
        if len(parts) >= 3:
            try:
                ply_num = int(parts[-1])  # Last number is ply index
                ply_meshes[ply_num - 1] = msh_file  # Convert to 0-indexed
            except ValueError:
                continue
    
    return ply_meshes


def estimate_mesh_resolution(nodes: np.ndarray, elements: np.ndarray) -> str:
    """
    Estimate mesh resolution from element sizes.
    
    Args:
        nodes: Node coordinates
        elements: Element connectivity
        
    Returns:
        Resolution estimate (e.g., "0.5mm", "1.0mm")
    """
    # Calculate element edge lengths
    if len(elements) == 0:
        return "unknown"
    
    # Sample some elements
    sample_size = min(1000, len(elements))
    sample_indices = np.random.choice(len(elements), sample_size, replace=False)
    sample_elements = elements[sample_indices]
    
    edge_lengths = []
    for elem in sample_elements:
        v0, v1, v2 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2),
        ]
        edge_lengths.extend(edges)
    
    mean_edge_length = np.mean(edge_lengths)
    
    # Estimate resolution based on typical element size
    if mean_edge_length < 0.75:
        return "0.5mm"
    elif mean_edge_length < 1.5:
        return "1.0mm"
    elif mean_edge_length < 3.0:
        return "2.0mm"
    else:
        return f"{mean_edge_length:.1f}mm"

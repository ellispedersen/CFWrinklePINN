# -*- coding: utf-8 -*-
"""

version 1.0
data: 2024-07-24

Reads nodes and elements of a specific element group 'elemGrNrsToExport[i]' from a binary AniForm afm mesh file 'filename'. 'elemGrNrsAll' is an array containing all the element group numbers (IDs) that are present in the afm file.

Nodes and Elements are both a dictionary with keys corresponding with elemGrNrsToExport, or for all element groups (elemGrNrsToExport=[]).

Example:

Read nodes and elements for all element groups and displaying progress:
Nodes, Elements, elemGrNrsAll = ReadAFMesh('model.afm', [], False) OR Nodes, Elements, elemGrNrsAll = ReadAFMesh('model.afm', silent = False)

Read nodes and elements for element groups 3 and 7, and show no progress:
Nodes, Elements, elemGrNrsAll = ReadAFMesh('model.afm', [3,7])

Read nodes and elements for one element group 3, show no progress:
Nodes, Elements, elemGrNrsAll = ReadAFMesh('model.afm', [3])


"""

import numpy as np

def ReadAFMesh(filename, elemGrNrsToExport = [], silent = True): 

    Nodes = dict();
    Elements = dict();
    
    with open(filename, 'rb') as fid:
    
    
        #%% Read the header of the file
        info = fid.read(32)
    
        if not silent:
            print(' ')
            print(info)
       
        # uint32: data format version
        data_format, = np.frombuffer(fid.read(4), dtype=np.uint32)
        # 28 bytes of empty space
        checkZeros = np.frombuffer(fid.read(28), dtype=np.uint8)
        if any(checkZeros):
            print('Reserved 28 bytes block contains non-zeros, whereas must contain solely zeros.')
            
            
        #%% Read the mesh groups
            
        starts = [64]
        elemGrNrsAll = []
        elemGrNr_buffer = fid.read(4)
        
        while elemGrNr_buffer != b'':
                   
            # uint32: id of tool part or laminate layer
            elemGrNr, = np.frombuffer(elemGrNr_buffer, dtype=np.uint32)
            elemGrNrsAll += [elemGrNr]
    
            # uint32: start of next group in bytes counting from start of file, 0 if last
            start_next_group, = np.frombuffer(fid.read(4), dtype=np.uint32)
            starts += [start_next_group]
                       
            exportGroup = (not(elemGrNrsToExport)) or (elemGrNr in elemGrNrsToExport)
            if exportGroup:
                
                # uint32: number of nodes
                nnodes, = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32: number of elements
                nelem, = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32: start of element data of this group in bytes counting from start of file
                start_elemdata, = np.frombuffer(fid.read(4), dtype=np.uint32)
                # # uint32: element format (1: triangle)
                elem_format, = np.frombuffer(fid.read(4), dtype=np.uint32)
                
                # for element format
                #  1 = Triangle,  3 Nodes, 3D
                #  2 = Tet,       4 Nodes, 3D
                #  3 = Tet,      10 Nodes, 3D
                #  4 = Point,     1 Node,  3D
                #  5 = Truss,     2 Nodes, 3D
                #  6 = Quad,      4 Nodes, 3D
                #  7 = Hex,       8 Nodes, 3D
                #  8 = Hex,      20 Nodes, 3D
                #  9 = Triangle,  6 Nodes, 3D
                # 10 = Quad,      8 Nodes, 3D
                # 11 = Quad,      9 Nodes, 3D
                # 12 = Hex,      27 Nodes, 3D
                # 13 = Wedge,     6 Nodes, 3D
                # 14 = Wedge,    15 Nodes, 3D
                # 15 = Pyramid,   5 Nodes, 3D
                # 16 = Pyramid,  13 Nodes, 3D
                # 17 = WedgeLQ,   9 Nodes, 3D
                def f(elem_format):
                    return {
                        1 : 3,
                        2 : 4,
                        3 : 10,
                        4 : 1,
                        5 : 2,
                        6 : 4,
                        7 : 8,
                        8 : 20,
                        9 : 6,
                        10 : 8,
                        11 : 9,
                        12 : 27,
                        13 : 6,
                        14 : 15,
                        15 : 5,
                        16 : 13,
                        17 : 9
                        }[elem_format]

                nodes_per_el = f(elem_format)
                
                # uint32: node format (1: float32)
                node_format, = np.frombuffer(fid.read(4), dtype=np.uint32)
                # 28 bytes reserved for future use, must be 0.
                checkZeros = np.frombuffer(fid.read(28), dtype=np.uint8)
                if any(checkZeros):
                    print('Reserved 28 bytes block contains non-zeros, whereas must contain solely zeros.')

                # 4 byte uint32: simID
                sim_id, = np.frombuffer(fid.read(4), dtype=np.uint32)
                # 4 byte uint32 -1(0xffffffff), control number
                control_number = hex(np.frombuffer(fid.read(4), dtype=np.uint32)[0])
                # The header is followed by data:
                # followed by node data:
                # for node format 1:
                # number of nodes*3*float32: xyz coordinates of the nodes
                NCoord = np.frombuffer(fid.read(4*nnodes*3), dtype=np.float32, count = nnodes*3).reshape(nnodes,3)
                # number of nodes*uint32: external node number
                Nnrs = np.frombuffer(fid.read(4*nnodes), dtype=np.uint32, count=nnodes)
                
                Nodes[elemGrNr] = np.hstack((Nnrs[:,np.newaxis], NCoord))
        
                
                EIndx = np.frombuffer(fid.read(4*nelem*nodes_per_el), dtype=np.uint32, count = nelem*nodes_per_el).reshape(nelem, nodes_per_el)
                # EIndx = reshape(EIndx',nodes_per_el,[])';
                
                # number of elements*uint32: external element number
                Enrs = np.frombuffer(fid.read(4*nelem), dtype=np.uint32)
                
                Elements[elemGrNr] = np.hstack((Enrs[:,np.newaxis], EIndx))
        
                if not silent:
                    print('Read ', str(len(Enrs)), ' element(s) and ', str(len(Nnrs)), ' nodes into workspace for element group id: ', str(elemGrNr))
                
                # set pointer back
                fid.seek(fid.tell(),0)
                elemGrNr_buffer = fid.read(4)
                
            else:
                # set pointer to next group
                fid.seek(start_next_group,0)
                elemGrNr_buffer = fid.read(4)
                   
    return Nodes, Elements, elemGrNrsAll

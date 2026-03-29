# -*- coding: utf-8 -*-
"""

version 1.0
data: 2024-07-24

Reads a result of a specific element group 'elemGrNrs' and increment 'IncsToExport' from a binary AniForm afr result file 'filename'. If 'elemGrNrs' is empty, all groups are read. If 'IncsToExport' is empty, all increments are read.

When silent is true, header information will not be shown (default).

ResultsIncr is a nested dictionary which contains the result of the group 'Groups[i]' for the simulation per increment/step Increments[i] as denoted by the key : ResultsIncr[Increment[i]][Groups[i]]

res_type is defined as:
Result type (res_type)
  0 = UnKnownT
 10 = ScalarT	    1 float    scalar type				            s
 20 = VectorT	    3 floats   vector type				            vx,vy,vz	
 30 = TensorUNIT    1 floats   uniaxial tensor type	                txx
 31 = STensorPSST   3 floats   symmetric plain stress tensor type	txx,tyy,txy
 32 = TensorPSST    4 floats   plain stress tensor type	            txx,tyy,txy,tyx
 33 = STensorPSNT   4 floats   symmetric plain strain tensor type	txx,tyy,tzz,txy
 34 = TensorPSNT    5 floats   plain strain tensor type	            txx,tyy,tzz,txy,tyx  
 35 = STensorT      6 floats   symmetric tensor type			    txx,tyy,tzz,txy,tyz,txz
 36 = TensorT       9 floats   non-sym tensor type                  txx,tyy,tzz,txy,tyz,txz,tyx,tzy,tzx
 40 = EulerT	    3 floats   Euler angle                          rx,ry,rz
 
indices_included is a bool: If True: indices included,
                            If False: no indices and results are given for all nodes
 
res_id is defined in the PrePost documentation.

The 'filename' characters relate to the result ID and subID:
fname_x_y.afr
x     =  RES ID
y     =  subresult id
fname = base filename of the simulator inputfile

Examples:

Read all element groups and increments:
ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult('model_40_1.afr')

Read for element groups 10 and 12, increments 5 and 10, and show header information in the console:
ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult('model_105_1.afr', elemGrNrs = [10,12], IncsToExport = [5,10], silent = False)


"""

import numpy as np

def ReadAFResult(filename, elemGrNrs = [], IncsToExport = [], silent = True):

    ResultsIncr = dict()
    Groups = []
    Increments = []
    res_type = []
    indices_included = False
    res_id = []

    with open(filename, 'rb') as fid:
    
        # read the header of the file
        info = fid.read(32)
        
        if not silent:
            print(' ')
            print(info)
        
        # uint32: data format version
        data_version, = np.frombuffer(fid.read(4), dtype = np.uint32)
        # uint32: result id
        res_id, = np.frombuffer(fid.read(4), dtype = np.uint32)
        
        # uint32: subresult id
        subres_id, = np.frombuffer(fid.read(4), dtype = np.uint32)
        # uint32: start of next results group in bytes counting from start of file
        start_next_group, = np.frombuffer(fid.read(4), dtype = np.uint32)
        starts = start_next_group
        # 8 bytes of empty space
        checkZeros = np.frombuffer(fid.read(8), dtype = np.uint8)
        if any(checkZeros) and silent is not True:
            print('Reserved 8 bytes block contains non-zeros, whereas must contain solely zeros.')
        
        # 4 bytes uint32: simID
        simID, = np.frombuffer(fid.read(4), dtype = np.uint32)
        # 4 bytes uint32 -1(0xffffffff), control number
        control_number = hex(np.frombuffer(fid.read(4), dtype=np.uint32)[0])
        # 4 bytes  uint32:   length of name
        name_length = np.frombuffer(fid.read(4), dtype = np.uint32)[0]
        name = fid.read(name_length)
        
        if not silent:
            print(' ')
            print('----------------------- General file header ------------------------------')
            print(info.decode('ascii').strip(), ', version ', str(data_version))
            print('Result id: ', str(res_id), ' Subresult id: ', str(subres_id), ' Name: ', name.decode('ascii'), ', Control nr: ', str(control_number))
        
        
        
        #%% read the result groups
        
        iresGr = 1
        ResultsIncr = dict()
        Groups = []
        Increments = []
        
        data_buffer = fid.read(12)
        
        while data_buffer != b'':
            # Read at once:
            # uint32: id of tool part or laminate layer
            # uint32: increment number
            # uint32: start of next results group in bytes counting from start of file
            data = np.frombuffer(data_buffer, dtype = np.uint32)
        
            id = data[0]
            incr_nr = data[1]
            if incr_nr not in Increments:
                ResultsIncr[incr_nr] = dict()
            start_next_group = data[2]
                        
            exportGroup = (not(elemGrNrs)) or (id in elemGrNrs)
            exportIncr = not(IncsToExport) or (incr_nr in IncsToExport)
                       
            if exportGroup and exportIncr:
                
                # uint32: result type
                res_type = np.frombuffer(fid.read(4), dtype = np.uint32)[0]
                # uint32: results format
                res_format = np.frombuffer(fid.read(4), dtype = np.uint32)[0]
                # uint32: number of results. Does not have to be equal to nr of nodes in mesh      
                n_res = np.frombuffer(fid.read(4), dtype = np.uint32)[0]
                # bool: indicesincluded. If true: indices included, if false: no indices and results are given for all nodes
                indices_included = np.frombuffer(fid.read(1), dtype = bool)[0]
                if indices_included: # check whether current approach holds for True and whether conversion is needed
                    indices_included = True
                else:
                    indices_included = False
                
                # 31 bytes reserved for future use, must be 0.
                checkZeros = np.frombuffer(fid.read(31), dtype = np.uint8)
                # checkZeros = np.frombuffer(fid.read(31), dtype = np.uint8)
                if any(checkZeros) and silent is not True:
                    print('Reserved 31 bytes block contains non-zeros, whereas must contain solely zeros.')
        
                # 4 bytes uint32: simID
                simID, = np.frombuffer(fid.read(4), dtype = np.uint32)
                # 4 bytes uint32 -1(0xffffffff), control number
                control_number = hex(np.frombuffer(fid.read(4), dtype=np.uint32)[0])
        
                # The result group header is followed by the data    
                #     4 b  uint32:   length of name
                #     n b  nchar:    result name
                #     number of nodes * uint32:					node index
                #     number of results * sizeof(result type)*sizeof(result format):  results
                if not silent:
                    print(' ')
                    print('------------------------ Results group ------------------------------')    
                    print('Id:', str(id), ', incr:', str(incr_nr), ', result type:', str(res_type), ', result format:', str(res_format))
                    if indices_included:
                        word = 'True'
                    else:
                        word = 'False'
                    print('Indices included:', word, ', control nr:', str(control_number))
                
                def f(res_type):
                    return {
                        10 : 1,
                        20 : 3,
                        30 : 1,
                        31 : 3,
                        32 : 4,
                        33 : 4,
                        34 : 5,
                        35 : 6,
                        36 : 9,
                        40 : 3
                        }[res_type]

                n_floats = f(res_type)       
                
                Results = np.zeros([n_res, n_floats+1])
        
                # number of nodes * uint32: node index
                if indices_included:
                    Results[:,0] = np.frombuffer(fid.read(n_res*4), dtype = np.uint32)
                
                # number of results * sizeof(result type)*sizeof(result format):  results
                R = np.frombuffer(fid.read(n_floats*n_res*4), dtype = np.float32)
                Results[:,1:n_floats+1] = np.reshape(R, [-1,n_floats])
                
                # store result in workspace
                ResultsIncr[incr_nr][id] = Results
                Groups += [id]
                Increments += [incr_nr]
                iresGr += 1
                
                data_buffer = fid.read(12)
                
            else:
                # set pointer to next group
                fid.seek(start_next_group,0)
                data_buffer = fid.read(12)
    
        return ResultsIncr, Groups, Increments, res_type, indices_included, res_id


# -*- coding: utf-8 -*-
"""

version 1.0
data: 2024-07-24

Reads the binary AniForm Solution (AFS) file

Returns:
  Name            simulation name, as read from the afi file
  Description     simulation description, as read from the afi file
  IncrementInfo   contains the increment info (v2 and higher)
  Version         version of the afs
  Core            core information, block 3 (v3 and higher)
  SimulationInfo  simulation info, block 4 (v3 and higher)
  Timing          contains the timing info, block 5 (v3 and higher)
  
General solution file header:
     |  32 b   char:   "Aniform solution data v3        "
 64b |   4 b   uint32: data format version
     |  20 b   reserved for future use, must be 0   
     |   4 b   uint32: unique simulation identifier           
     |   4 b   uint32 -1(0xffffffff), control number     
     Followed by blocks
         |   4 b   uint32: block type nr
         |   4 b   uint32: length of block information in bytes
         Block information

NOTE:  This format has been designed to be forward compatible.
       Skip unknown blocks!
NOTE2: afs v2 files exist where blockinfo size of block nr 2 is 
       incorrectly exported as 71 bytes
         
Block type nr 1, simulation information, only one
   |   4 b  uint32:   length of name
   |   n b  char:     name
   |   4 b  uint:     length of description
   |   n b  char:     description

Block type nr 2, increment info (63 bytes)
   |   4 b  uint32:   incr nr
   |   8 b  double:   time at end of increment
   |   4 b  uint32:   nr of iterations  
   |   8 b  double:   norm unbalance
   |   8 b  double:   norm displacement
   |   1 b  bool:     has converged
   |   4 b  uint32:   current loadblock
   |   4 b  uint32:   nr of loadblocks
   |   8 b  double:   progress current loadblock
   |   8 b  double:   next increment size
   |   4 b  uint32:   nr of increments to go
   |   1 b  bool:     restartfile written
   |   1 b  bool:     resultfiles written

Block type nr 3, core info (58 bytes), only one
   |   4 b  uint32:   major version nr
   |   4 b  uint32:   minor version nr
   |   4 b  uint32:   build version nr
   |   1 b  uint8:    product status, 0=release, 1=rc, 2=beta
   |   1 b  uint8:    extension set, 0=no, 1=avx2
   |   1 b  bool:     parallel version
   |   1 b  bool:     cluster version
   |   1 b  bool:     debug build
   |  40 b  uint8:    40 byte hash
   |   1 b  uint8:    platform, 0=win, 1=linux
   
Block type nr 4, simulation info (40 bytes), only one
   |   8 b  uint64:   starting time in ISO C time_t (elapsed s since 1-1-1970 UTC)
   |  32 b  uint8:    MD5 checksum afi file
   
Block type nr 5, increment timing info (32 bytes)
   |   4 b  uint32:   current loadblock
   |   4 b  uint32:   incr nr
   |   8 b  double:   wall time
   |   8 b  double:   wall time current block
   |   8 b  double:   wall time current increment    
 
  
Example:
    
Read the simulation name, description and information per increment
Name, Description, IncrementInfo, Version, Core, SimulationInfo, Timing = ReadAFSFile('model.afs')

"""

import numpy as np
import pandas as pd
import datetime
import sys

def ReadAFSFile(filename, silent = True):
    
    with open(filename, 'rb') as fid:
        
        #%% Read the header of the file
        info = fid.read(32)
        
        if info[:21] != b'Aniform solution data':
            print('This is not an AniForm Solution file.')
            fid.close()
            sys.exit()
        
        if not silent:
            print('Reading .afs file ...')
            print(info.decode('ascii'))
        
        #% uint32: data format version
        Version, = np.frombuffer(fid.read(4), dtype=np.uint32)
        # 20 bytes of empty space
        fid.seek(fid.tell()+20,0)
        # uint32: unique simulation identifier
        sim_id = np.frombuffer(fid.read(4), dtype=np.uint32)
        # uint32 -1(0xffffffff), control number
        control_number = hex(np.frombuffer(fid.read(4), dtype=np.uint32)[0])
        if control_number != '0xffffffff':
            print('File corrupted. Control number incorrect')
            
        #%% Read the blocks
        
        Name = ''
        Description = ''
        IncrementInfo = pd.DataFrame(index = ['incr_nr', 't_end', 'nr_iterations', 'norm_unbalance', 'norm_displacement', 
                                                 'has_converged', 'current_loadblock', 'nr_of_loadblocks', 'progress_current_loadblock', 
                                                 'next_increment_size', 'nr_incr_to_go', 'restartfile_written', 'resultsfile_written'])
        Core = pd.DataFrame(index = ['major', 'minor', 'build', 'status', 'extension_set', 'parallel', 'cluster', 'debug_build', 'hash', 'platform'])
        SimulationInfo = pd.DataFrame(index = ['starting_datetime', 'afichecksum'])
        Timing = pd.DataFrame(index = ['current_loadblock', 'incr_nr', 'wall_time', 'wall_time_block', 'wall_time_incr'])
        
        block_buffer = fid.read(4)
        
        while block_buffer != b'':
            block_nr = np.frombuffer(block_buffer, dtype=np.uint32)
            block_length = np.frombuffer(fid.read(4), dtype=np.uint32)
            start = fid.tell()
            
            if block_nr == 1:
                # uint32:   length of name
                name_length = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                # n b  char:     name
                Name = fid.read(name_length).decode('ascii')
                # uint:     length of description
                descr_length = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                # n b  char:     description
                Description = fid.read(descr_length).decode('ascii')
            
            elif block_nr == 2:
                
                data_increment_info = pd.Series(dtype='object')
                # uint32:   incr nr
                data_increment_info['incr_nr'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # double:   time at end of increment
                data_increment_info['t_end'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # uint32:   nr of iterations
                data_increment_info['nr_iterations'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # double:   norm unbalance
                data_increment_info['norm_unbalance'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # double:   norm displacement
                data_increment_info['norm_displacement'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # bool:     has converged
                data_increment_info['has_converged'], = np.frombuffer(fid.read(1), dtype=bool)
                # uint32:   current loadblock
                data_increment_info['current_loadblock'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32:   nr of loadblocks
                data_increment_info['nr_of_loadblocks'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # double:   progress current loadblock
                data_increment_info['progress_current_loadblock'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # double:   next increment size
                data_increment_info['next_increment_size'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # uint32:   nr of increments to go
                data_increment_info['nr_incr_to_go'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # bool:     restartfile written
                data_increment_info['restartfile_written'], = np.frombuffer(fid.read(1), dtype=bool)
                # bool:     resultfiles written
                data_increment_info['resultsfile_written'], = np.frombuffer(fid.read(1), dtype=bool)
                
                IncrementInfo = pd.concat((IncrementInfo, data_increment_info), axis=1)
           
            elif block_nr == 3:
                
                data_core = pd.Series(dtype='object')
                # uint32:   major version
                data_core['major'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32:   minor version
                data_core['minor'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32:   build nr
                data_core['build'], = np.frombuffer(fid.read(4), dtype=np.uint32)     
                # uint8:    status
                data_core['status'], = np.frombuffer(fid.read(1), dtype=np.uint8)
                # uint8:    extension set
                data_core['extension_set'], = np.frombuffer(fid.read(1), dtype=np.uint8)
                # bool:     parallel edition
                data_core['parallel'], = np.frombuffer(fid.read(1), dtype=bool)
                # bool:     cluster edition
                data_core['cluster'], = np.frombuffer(fid.read(1), dtype=bool)
                # bool:     debug build
                data_core['debug_build'], = np.frombuffer(fid.read(1), dtype=bool)
                # uint8[40]:  git hash
                hashbytes = np.frombuffer(fid.read(40), dtype=np.uint8)
                data_core['hash'] = ''.join([chr(val) for val in hashbytes[hashbytes > 0]])
                # uint8:    platform
                data_core['platform'], = np.frombuffer(fid.read(1), dtype=np.uint8)
                
                Core = pd.concat((Core, data_core), axis=1)
                
            elif block_nr == 4:
                
                data_simulation_info = pd.Series(dtype='object')
                
                # uint64:   starting time
                elapsed = np.frombuffer(fid.read(8), dtype=np.uint64)
                data_simulation_info['starting_datetime'] = datetime.datetime.fromtimestamp(
                    int(elapsed[0]), datetime.UTC
                ).strftime("%d-%b-%Y %H:%M:%S")
                # uint8[16]:  afi md5 hash
                hashbytes = np.frombuffer(fid.read(32), dtype=np.uint8)
                data_simulation_info['afichecksum'] = ''.join([chr(val) for val in hashbytes[hashbytes > 0]])
                
                SimulationInfo = pd.concat((SimulationInfo, data_simulation_info), axis=1)
                
            elif block_nr == 5:

                data_timing = pd.Series(dtype='object')
                
                # uint32:   current loadblock
                data_timing['current_loadblock'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # uint32:   incr nr                
                data_timing['incr_nr'], = np.frombuffer(fid.read(4), dtype=np.uint32)
                # double:   wall time 
                data_timing['wall_time'], = np.frombuffer(fid.read(8), dtype=np.float64)
                # double:   wall time block
                data_timing['wall_time_block'], = np.frombuffer(fid.read(8), dtype=np.float64) 
                # double:   wall time increment
                data_timing['wall_time_incr'], = np.frombuffer(fid.read(8), dtype=np.float64)

                Timing = pd.concat((Timing, data_timing), axis=1)
                
            else:
                 print('Skipping unknown block nr ' + str(block_nr[0]) + ' of length ' + str(block_length[0]))
                 for i in range(len(block_length)):
                     # check for end of file
                     if fid.read(1) == b'':
                         break
            
            if block_nr == 2 and block_length == 71:
                # there are versions with an incorrect block length written for block nr 2
                block_length = 63
    
            bytesRead = fid.tell() - start
            if bytesRead != block_length:
                print('Incorrect block length. Block ' + str(block_nr[0]) + ' length ' + str(block_length[0]) + ', but ' + str(bytesRead[0]) + ' bytes read.')
                break
            
            block_buffer = fid.read(4)
        
        if not silent:
            print('Reading .afs file finished.')
        
    return Name, Description, IncrementInfo, Version, Core, SimulationInfo, Timing

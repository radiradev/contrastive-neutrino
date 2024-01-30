#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
OSFAPI_DIR = os.path.dirname(os.path.abspath(__file__))
OSFAPI_DIR = os.path.dirname(OSFAPI_DIR)
sys.path.insert(0, OSFAPI_DIR)
                    
import sys,argparse
import numpy as np
from osf.analysis_apis import data_reader, parse_particle, csv_writer

def make_particle_csv(product_name,start,num,output_file,input_files):
    
    data_in = data_reader()
    data_in.add_data(product_name)
    for f in input_files: data_in.add_file(f)

    data_out = csv_writer(output_file)

    if num < 0:
        num = data_in.entry_count()

    end = start + num
    
    for entry in np.arange(start,end):

        data_in.read(entry)
        data = data_in.data(product_name)
        part_info = parse_particle(data)

        if entry == start:
            data_out.header(part_info.keys())

        data_out.write(part_info.values())

        sys.stdout.write('%d%% done..\r' % int(float(entry-start)/float(end-start)*100.))
        sys.stdout.flush()

    data_out.close()
    sys.stdout.write('\n')

def main():
    """
    A main function to be executed
    """

    DEFAULT_LABEL = 'particle_mcst'
    DEFAULT_START =  0
    DEFAULT_NUM   = -1
    DEFAULT_OUTPUT= 'ana.csv'
    
    parser = argparse.ArgumentParser(description='Michel/Delta-ray label correction script')
    parser.add_argument('-of','--output_file',type=str,default=DEFAULT_OUTPUT,
                        help='Output file name [default: %s]' % DEFAULT_OUTPUT)
    parser.add_argument('-il','--input_label',type=str,default=DEFAULT_LABEL,
                        help='Input data product label [default: %s]' % DEFAULT_LABEL)
    parser.add_argument('-s','--start',type=int,default=DEFAULT_START,
                        help='Start entry [default: %d]' % DEFAULT_START)
    parser.add_argument('-n','--num',type=int,default=DEFAULT_NUM,
                        help='Number of entries to process (value<0 means "all") [default: %d]' % DEFAULT_NUM)
    parser.add_argument('input_files', nargs='*', help='List input files (space separated)')
    args = parser.parse_args()
    
    if len(args.input_files) < 1:
        print('Error: need to provide input files (>0, non provided)')
        print('Try --help.')
        return 1

    make_particle_csv( product_name = args.input_label,
                       start = args.start,
                       num = args.num,
                       output_file = args.output_file,
                       input_files = args.input_files )

if __name__ == '__main__':
    main()

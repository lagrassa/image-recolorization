#import matlab.engine
import sys
import os
import pdb
import NNcolorize


def main(filename):
    #First run neural nets on the image
    NNcolorize.main(filename, True)
    command = 'matlab -r ../matlab/run_recolorization.m "'+filename+'";'+ 'quit force'
    os.system(command)
    #eng = matlab.engine.start_matlab()
    #eng.SegmentColorize(nargout=0)

if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)




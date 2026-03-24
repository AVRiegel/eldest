#!/usr/bin/python
##########################################################################
#                          Create ELDEST movie                           #
##########################################################################
# written by: Alexander Riegel, May 2023                                 #
##########################################################################

import argparse
import os
from contextlib import contextmanager
import subprocess
import shutil


# set up argument parser
parser = argparse.ArgumentParser(
        description='''ELDEST -- cinema.py :
        An auxiliary programme to create a movie (gif) of P-E_kin spectra for different times
        from the file movie.dat (produced by, e.g., nuclear_dyn.py .''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='By Alexander V. Riegel.')
parser.add_argument('-E', '--E_lims', nargs=2, default=[9.8, 10.5], metavar=('E_low', 'E_high'),
                    help='Lower and upper limit for E_kin (in eV).')
args = parser.parse_args()



@contextmanager # https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory/24176022#24176022 (2023-May-22)
def cd(newdir):
    """Context manager for changing the current working directory"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

gnufile = open('gnufile.gp', 'w')
gnufile.write("""
        set terminal png size 1600,1200 enhanced
        set xrange [ARG2:ARG3]
        set yrange [0:1.1*ARG1]             # set ymax as 110 % of maximum intensity (will be passed as argument)
        print ARG1
        print ARG2
        print ARG3
        set xlabel "E_{kin} / eV"
        set ylabel "P(E_{kin}) / a.u."
        set key autotitle columnhead
        files = system('ls -1 *.txt')
        do for [file in files]{
        set output file.".png"
        plot file u 1:(column($#)) tit columnhead(1) w l lw 3 lc 3    # $# evaluates to total number of columns in current line
        }
        """)
gnufile.close()


os.mkdir('tempplot')
os.system('cp movie.dat tempplot/.')
os.system('mv gnufile.gp tempplot/.')
maxim = float(subprocess.check_output("awk 'BEGIN{a=0}{if ($NF>0+a && NF>2) a=$NF} END{print a}' movie.dat", shell=True)[:-1])    # maximum intensity; NF>2 skips the "xxx fs" lines

with cd('./tempplot'):    # automatically reverts back to cwd after being finished
    # 1) split movie.dat along empty lines, enumerate files with four-digit number (FILE0001.txt, FILE0002.txt etc.)
    # 2) gnuplot, pass maximum intensity as variable
    # 3) ffmpeg converts png images into gif movie
    os.system("""   tr -d '\r' < movie.dat | awk '{print > sprintf("%s%04d%s", "FILE", ++CNT, ".txt")}' RS=''   """)
    os.system("gnuplot -c gnufile.gp {} {} {}".format(maxim, *args.E_lims))
    os.system("ffmpeg -loglevel warning -f image2 -r 10.0 -i FILE%04d.txt.png -q:v 1 -pattern_type globe -codec gif ../movie.gif")

shutil.rmtree("tempplot")

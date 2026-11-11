#!/bin/sh

### Job name
#PBS -N p6_md_y

### Output files
#PBS -o p6_md.log
#PBS -j oe

### Number of requested memory,etc.
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=40Gb:vmem=40Gb
#PBS -l cput=540000:00:00
#PBS -l walltime=540000:00:00

### Job is not rerunable
#PBS -r n
### A mail is sent after end or abort of the job
#PBS -m ae

##### EDIT HERE to change host (mulde or digedag), directory in which
#####           the executable is located, name of the executable and
#####           name of the outputfile, additionally the path to ssh
#####           is put to a variable

HOST=$PBS_O_HOST
SSH="/usr/bin/ssh"
PYTHONPATH="$HOME/.local/lib/python2.7/site-packages"
PBS_O_EXEDIR="$PBS_O_WORKDIR"
TMP_DIR="/scratch/$USER/p6_$PBS_JOBID"
EXECUTABLE="md.py"
INFILE=" "
OUTFILE="out_cont2"

#### END OF EDITABLE LINES

# useful information for debugging
echo Running on host `hostname`
echo Time is `date`
ulimit -l unlimited
echo ulimit: $(ulimit -l)

### Copy application directory on compute node

set +x 

[ -d $PBS_O_EXEDIR ]  || mkdir -p $PBS_O_EXEDIR
[ -d $PBS_O_WORKDIR ] || mkdir -p $PBS_O_WORKDIR
[ -d $PYTHONPATH ]    || mkdir -p $PYTHONPATH


# Copying executables and data to the nodes

[ -w $PBS_O_EXEDIR ] && \
rsync -avz --rsh=$SSH $HOST:$PBS_O_EXEDIR `dirname $PBS_O_EXEDIR`
rsync -avz --rsh=$SSH $HOST:$PYTHONPATH `dirname $PYTHONPATH`
rsync -avz --rsh=$SSH $HOST:$PBS_O_WORKDIR `dirname $PBS_O_WORKDIR`


# Copy all data into a local scratch directory
cd $PBS_O_WORKDIR
ls


# Save the jobid in the outfile
echo "PBS-JOB-ID was $PBS_JOBID" > $OUTFILE

# Run the executable
export PATH=$PBS_O_EXEDIR:/opt/openmpi/bin:$PYTHONPATH:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH

$PBS_O_EXEDIR/$EXECUTABLE $INFILE > $OUTFILE

# SAVE INFO ABOUT USED RESOUCES IN FILE resources-$PBS_JOBID 
echo "COMMANDLINE: $PBS_O_EXEDIR/$EXECUTABLE >> $OUTFILE" > resources-$PBS_JOBID
echo "Calculation was done in directory $PBS_O_WORKDIR" >> resources-$PBS_JOBID

$SSH $HOST /opt/pbs/bin/qstat -f $PBS_JOBID >> resources-$PBS_JOBID

# Removing temporary files is forced.

ls tmp* >/dev/null 2>&1 && rm $PBS_O_WORKDIR/tmp*

#rsync -avz --rsh=$SSH $TMP_DIR/* $HOST:$PBS_O_WORKDIR &&  /bin/rm -rf $TMP_DIR




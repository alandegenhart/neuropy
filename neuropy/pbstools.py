__version__ = '0.1.0'

import argparse
import copy
import subprocess as sp
import logging
import os

logger = logging.getLogger(__name__)
default_settings = {
    'jobname': 'Default_Job_Name',
    'email': 'alan.degenhart@alleninstitute.org',
    'email_options': 'a',
    'queue': 'braintv',
    'mem': '20g',
    'walltime': '01:00:00',
    'ncpus': None,
    'gpus': None,
    'ppn': 1,
    'nodes': 1,
    'jobdir': None,
    'outfile': '$PBS_JOBID.out',
    'errfile': '$PBS_JOBID.err',
    'environment': {},
    'array': None,
    'priority': None,
    'rerunable': False
}


class Deal(list):
    pass


class RemoteSweep(list):

    def __str__(self):
        return 'RemoteSweep(%s)' % super(list, self).__str__()


class PBSJob(object):

    qsub_command = 'qsub'

    def __init__(self, command, **kwargs):
        # Set object attributes
        self.command = command

        # TODO: move settings in here? Not sure why they are defined above

        # Add job settings to object attributes
        settings_dict = copy.copy(default_settings)
        settings_dict.update(kwargs)
        for key, val in settings_dict.items():
            setattr(self, key, val)

    def run(self, verbose=True, dryrun=False):

        if not dryrun:
            #
            sub_process = sp.Popen(PBSJob.qsub_command,
                                   shell=True,
                                   stdin=sp.PIPE,
                                   stdout=sp.PIPE,
                                   close_fds=True,
                                   encoding='utf8')
            sp_output = sub_process.stdout
            sp_input = sub_process.stdin

            if sp_input is None:
                raise Exception('could not start job')

        # Define lines of submission script
        script_lines = []
        script_lines.append('#!/bin/bash\n')
        #
        if self.array is not None:
            script_lines.append(f'#PBS -t {self.array[0]}-{self.array[1]}\n')
        # Execution priority
        if self.priority is not None:
            try:
                assert self.priority in ['high', 'med', 'low']
            except AssertionError:
                raise ValueError('Priority "%s" is not "high/med/low"')

            script_lines.append(f'#PBS -W x=QOS:{self.priority}\n')

        # Queue
        script_lines.append(f'#PBS -q {self.queue}\n')
        # Job name -- this is the unique identifier for the job
        script_lines.append(f'#PBS -N {self.jobname}\n')
        # Email -- status messages are sent to this
        if self.email is not None:
            script_lines.append(f'#PBS -M {self.email}\n')
        # Email options -- specifies how messages should be sent. Messages can
        # be sent on job execution, completion, and error.
        if self.email_options is not None:
            script_lines.append(f'#PBS -m {self.email_options}\n')
        #
        if not self.rerunable:
            script_lines.append('#PBS -r n\n')

        # Define cpu/gpu/nodes/ppn specifications
        cpu_ppn_node_list = []
        # Number of nodes to request
        if self.nodes is not None:
            cpu_ppn_node_list.append(f'nodes={self.nodes}')
        # Number of CPUs to request
        if self.ncpus is not None:
            cpu_ppn_node_list.append(f'ncpus={self.ncpus}')
        # Number of GPUs to request
        if self.gpus is not None:
            cpu_ppn_node_list.append(f'gpus={self.gpus}')
        # Processors per node to request
        if self.ppn is not None:
            cpu_ppn_node_list.append(f'ppn={self.ppn}')
        # Construct full cpu/gpu/node/ppn string and add to commands
        script_lines.append(f'#PBS -l {":".join(cpu_ppn_node_list)}\n')

        # Define memory and wall time string
        mem_walltime = []
        # Memory
        if self.mem is not None:
            mem_walltime.append(f'mem={self.mem}')
        # Wall time
        if self.walltime is not None:
            mem_walltime.append(f'walltime={self.walltime}')
        # Construct complete memory/walltime string and add
        if len(mem_walltime) > 0:
            script_lines.append(f'#PBS -l {",".join(mem_walltime)}\n')

        # Add directory path
        if self.jobdir is not None:
            script_lines.append(f'#PBS -d {os.path.expanduser(self.jobdir)}\n')

        # Add output and error file paths
        script_lines.append(f'#PBS -o {os.path.expanduser(self.outfile)}\n')
        script_lines.append(f'#PBS -e {os.path.expanduser(self.errfile)}\n')


        env_list = []
        for variable, value in self.environment.items():
            env_list.append(f'{variable}={value}')

        if len(env_list) > 0:
            script_lines.append(f'#PBS -v {",".join(env_list)} \n')

        script_lines.append(f'{self.command}\n')

        script_string = ''.join(script_lines)
        logger.info(script_string)

        # Display message if performing a dry run
        if dryrun:
            print('[DRYRUN] Script will not be executed.\n')

        # Iterate over lines and write
        for line in script_lines:
            if verbose:
                # Display output to user
                print(line, end='')
            if not dryrun:
                sp_input.write(line)

        # Close
        if not dryrun:
            sp_input.close()
            result = sp_output.read()
            if verbose:
                print(result)
        else:
            result = None

        return result


class PythonJob(PBSJob):
    """Python PBS job."""

    def __init__(self, script, python_executable='python', conda_env=None, python_args='', python_path=None, **kwargs):

        self.python_executable = python_executable
        self.conda_env = conda_env
        assert os.path.exists(script)
        self.script = script
        self.python_args = python_args

        # If a conda environment is specified, add it
        command = ''
        if conda_env is not None:
            command += f'source activate {self.conda_env}\n'

        # Export python path if desired
        if python_path is not None:
            command += f'export PYTHONPATH={python_path}\n'

        # Specify python script command.  For a python job, the command will be
        # of the form: 'python script.py --arg_name arg_val'
        command += '%s %s %s' % (self.python_executable, self.script, self.python_args)

        # Call parent class to specify PBSJob parameters
        super(PythonJob, self).__init__(command, **kwargs)


def create_argument_string(arg_dict=None, flags=None):
    """Create an argument string for a python function.

    """

    arg_str = ''
    if arg_dict is not None:
        # Create key/value pair strings
        for k, v in arg_dict.items():
            arg_str += f' --{k} {v}'

    if flags is not None:
        # Flags don't have associated values, so just list them out
        for f in flags:
            arg_str += f' --{f}'

    return arg_str


if __name__ == "__main__":  # pragma: no cover
    pass


import argparse
import os
import queue
import sys

from collections import defaultdict

import radical.pilot as rp
import radical.utils as ru

# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT'] = 'TRUE'

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

RESOURCE_DESCRIPTION = {
    'resource': 'anl.polaris',
    'project': 'NNNNN',  # 'FoundEpidem'
    'nodes': 1,
    'runtime': 60
}

TASK_PRE_EXEC_ENV = [
    'module use /soft/modulefiles; module load conda',
    'eval "$(conda shell.posix hook)"',
    'conda activate cellseg'
]

NUM_PIPELINES = 1  # pipeline instances
IMAGES_PATH = ('/eagle/FoundEpidem/astroka/fib_and_htert/week_four/'
               '20241015_NewWeek4/20241015_ANL_CellPainting_W4P3_1__'
               '2024-10-15T16_29_06-Measurement1/Images')
SEG_IMAGE_TEMP = 'fib_rad_seg_temp'
IMAGE_TEMP = 'fib_rad_temp'
RESULTS = 'fib_rad'
WEEK = 'week_four'


class ExecManager:

    def __init__(self, resource_description, work_dir=None):
        self.tasks_finished_queue = queue.Queue()

        self._session = rp.Session()
        self._pmgr = rp.PilotManager(self._session)
        self._tmgr = rp.TaskManager(self._session)

        self._tmgr.register_callback(self.task_state_cb)

        # contains "radical.pilot.sandbox" with agent sandboxes per session
        resource_description['sandbox'] = os.path.abspath(work_dir or WORK_DIR)
        self._pilot = self._pmgr.submit_pilots(
            rp.PilotDescription(resource_description))

        self._tmgr.add_pilots(self._pilot)
        self._pilot.wait(rp.PMGR_ACTIVE)

    def close(self):
        self._session.close(download=True)

    def submit_tasks(self, *args, **kwargs):
        self._tmgr.submit_tasks(*args, **kwargs)

    def get_finished_task(self):
        output = None
        try:
            # task prefix (== pipeline name), task state
            output = self.tasks_finished_queue.get_nowait()
        except queue.Empty:
            pass
        return output

    def task_state_cb(self, task, state):
        if state not in rp.FINAL:
            # ignore all non-finished state transitions
            return
        prefix = task.uid.split('.', 1)[0]
        self.tasks_finished_queue.put([prefix, task.state])

    def generate_pipe_uid(self):
        return ru.generate_id('p%(item_counter)06d',
                              ru.ID_CUSTOM, ns=self._session.uid)

    def generate_task_uid(self, prefix, stage_id):
        prefix = prefix.replace('.', '_')
        return ru.generate_id(f'{prefix}.{stage_id}.%(item_counter)06d',
                              ru.ID_CUSTOM, ns=self._session.uid)


class Pipeline:

    def __init__(self, emgr, work_dir=None):
        self.emgr = emgr  # exec manager - will be isolated later?
        self.name = self.emgr.generate_pipe_uid()

        base_dir = os.path.abspath(work_dir or WORK_DIR)
        self.pipeline_dir = f'{base_dir}/JUMP_vision_model/rad_pipeline'
        self.work_dir = base_dir
        # TODO: make a `work_dir` to be pipeline-specific?
        #       self.work_dir = f'{base_dir}/{self.name}'

        self.stage_id = 0
        self.img_id = 0
        self.num_images = 0
        self.image_name = ''
        self.target_name = ''

    def submit_next(self):
        next_stage_id = self.stage_id + 1

        if next_stage_id == 2:
            # this is done once when switched from Stage 1 to 2
            self.num_images = ru.sh_callout(
                f'head -n 1 {self.pipeline_dir}/fib_rad_num_images.txt',
                shell=True)[0]

        elif next_stage_id == 3:
            self.image_name = ru.sh_callout(
                f'head -n 1 {self.pipeline_dir}/fib_rad_image_name.txt',
                shell=True)[0]
            self.target_name = ru.sh_callout(
                f'head -n 1 {self.pipeline_dir}/fib_rad_target_name.txt',
                shell=True)[0]

        elif next_stage_id == 5:
            if self.img_id < self.num_images:
                self.img_id += 1
                next_stage_id = 2

        try:
            submit_stage_func = getattr(self, f'stage_{next_stage_id}')
        except AttributeError:
            print(f'Pipeline {self.name} has finished')
            return 0

        self.stage_id = next_stage_id
        # submit tasks from the next stage
        return submit_stage_func()

    def stage_1(self):
        plate = 'fib_rad'
        treatment_file = f'{self.pipeline_dir}/week_four_fib_layout.xlsx'

        self.emgr.submit_tasks(rp.TaskDescription({
            'uid': self.emgr.generate_task_uid(prefix=self.name, stage_id=1),
            'executable': 'python',
            'arguments': [
                f'{self.pipeline_dir}/concat_images.py',
                '--image_path', IMAGES_PATH,
                '--plate', plate,
                '--treatment', treatment_file],
            'pre_exec':
                TASK_PRE_EXEC_ENV
                + [f'cd {self.work_dir}']
        }))
        return 1

    def stage_2(self):
        self.emgr.submit_tasks(rp.TaskDescription({
            'uid': self.emgr.generate_task_uid(prefix=self.name, stage_id=2),
            'executable': 'python',
            'arguments': [
                f'{self.pipeline_dir}/pull_images.py',
                '--index', str(self.img_id),
                '--path', IMAGES_PATH,
                '--temp', IMAGE_TEMP,
                '--seg', SEG_IMAGE_TEMP,
                '--res', RESULTS],
            'pre_exec':
                TASK_PRE_EXEC_ENV
                + [f'cd {self.work_dir}']
        }))
        return 1

    def stage_3(self):
        self.emgr.submit_tasks(rp.TaskDescription({
            'uid': self.emgr.generate_task_uid(prefix=self.name, stage_id=3),
            'executable': 'singularity run',
            'arguments': [
                'cellprofiler_4.2.6.sif', '-c', '-r',
                '-p', f'{self.pipeline_dir}/cropped_cells.cppipe',
                '-i', f'{self.pipeline_dir}/{IMAGE_TEMP}',
                '-o', f'{self.pipeline_dir}/{SEG_IMAGE_TEMP}/'
                      + f'{self.target_name}/'],
            'pre_exec':
                TASK_PRE_EXEC_ENV
                + ['module use /soft/spack/gcc/0.6.1/install/modulefiles/Core',
                   'module load apptainer',
                   f'cd {self.work_dir}']
        }))
        return 1

    def stage_4(self):
        self.emgr.submit_tasks(rp.TaskDescription({
            'uid': self.emgr.generate_task_uid(prefix=self.name, stage_id=4),
            'executable': 'python',
            'arguments': [
                f'{self.pipeline_dir}/change_names.py',
                '--target', self.target_name,
                '--name', self.image_name,
                '--src', SEG_IMAGE_TEMP,
                '--dst', RESULTS,
                '--week', WEEK],
            'pre_exec':
                TASK_PRE_EXEC_ENV
                + [f'cd {self.work_dir}']
        }))
        return 1


def get_args():
    parser = argparse.ArgumentParser(
        description='RADICAL-Pilot application for the Cell Painting Pipeline',
        usage='cell.rp.py [options]')
    parser.add_argument(
        '-w', '--work_dir',
        dest='work_dir',
        type=str,
        required=False)
    return parser.parse_args(sys.argv[1:])


def main():

    args = get_args()
    exec_mgr = ExecManager(resource_description=RESOURCE_DESCRIPTION,
                           work_dir=args.work_dir)

    # create pipelines
    pipes = {}
    for _ in range(NUM_PIPELINES):
        p = Pipeline(emgr=exec_mgr, work_dir=args.work_dir)
        pipes[p.name] = p

    # start executing pipelines (submit stages 1)
    tasks_active = defaultdict(int)
    for pipe_name, pipe in pipes.items():
        # start each pipeline
        tasks_active[pipe_name] += pipe.submit_next()  # num submitted tasks

    # loop to track the status of the executed tasks and to submit next stages
    while True:
        task_labels = exec_mgr.get_finished_task()
        if task_labels is None:
            # no finished tasks
            continue

        pipe_name, task_state = task_labels
        tasks_active[pipe_name] -= 1
        if tasks_active[pipe_name]:
            # if there were submitted a group of tasks within a stage,
            # and some of that tasks are still running
            continue

        tasks_active[pipe_name] += pipes[pipe_name].submit_next()

        # if there is no active tasks, then all pipelines finished
        if not sum(tasks_active.values()):
            break

    exec_mgr.close()


if __name__ == '__main__':
    main()


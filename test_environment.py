import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import requests

class Machine():
    def __init__(self, name="", address="", mem_total=0, cpu_total=0, slots=10):
        self.name = name
        self.address = address
        self.cpu_total = cpu_total
        self.mem_total = mem_total
        self.running_jobs = 0
        self.cpu = 0
        self.mem = 0
        self.mem_used = 0
        self.slots = slots

    def mem2slots(mem):
        return np.ceil(mem / self.mem_total)

    def schedule(job):
        return

    def snapshot():
        return 

class Job():
    def __init__(self, cpu_slots=0, mem_slots=0, image="", cmd="", time_slots=0):
        self.image = image
        self.cmd = cmd
        self.cpu_slots = cpu_slots
        self.mem_slots = mem_slots
        self.time_slots = time_slots

class ClusterEnvironment(py_environment.PyEnvironment):
    def __init__(self, job_queue_size, backlog_size, return_dict, horizon):
        self.returns = returns
        self.horizon = horizon
        self.machine = Machine(name="kimchi", address="http://kimchi.mocalab.org:3499", mem_total=180, cpu_total=24)
        self.job_queue_size = job_queue_size
        self.backlog_size = backlog_size

        # An action is an index from the job queue
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=job_queue_size,
            name='action'
        )

        # The state of the machine and cluster and job queue
        # Like 'images' in DeepRM
        machine_slots = self.machine.slots * 2
        job_slots = (machine_slots) * self.job_queue_size
        backlog_slots = self.backlog_size
        self.state_dim = (machine_slots + job_slots + backlog_slots, self.horizon)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.state_dim,
            dtype=np.int32,
            minimum=0,
            name='state'
        )
        
        self._state = np.zeros(state_dim)
        self._episode_ended = False
        self.current_time = 0

        # Detailed representation of state
        self.state_detail = [{
            "machine": {
                "running_jobs": 0,
                "cpu": 0,
                "mem": 0
            },
            "jobs": Job(),
            "backlog": 0
        } for i in range(self.horizon)]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros(state_dim)
        self._episode_ended = False
        return ts.restart(self._state)

env = ClusterEnvironment(5, 3, {}, 10)
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

    def observe(self):
        self.state["machine"]["running_jobs"] = self.machine.running_jobs
        self.state["machine"]["cpu"] = self.machine.cpu
        self.state["machine"]["mem"] = self.machine.mem
        self.state["backlog"] = len(self.backlog)

        compactState = self.state2vec(state)
        self._state = compactState
        return self.state, compactState

    def state2vec(self, state):
        vec = np.zeros(state_dim)

        machine = state["machine"]
        vec[self.job_queue_size][0] = np.float(machine["cpu"])
        vec[self.job_queue_size][1] = np.float(machine["mem"])

        for i in range(self.job_queue_size):
            if self.state["jobs"][i].image == "":
                vec[i][0] = 0.0
                vec[i][1] = 0.0
            else: 
                vec[i][0] = self.state["jobs"][i].cpu
                vec[i][1] = self.state["jobs"][i].mem

        vec[-1][0] = len(self.backlog)

        return vec

    def get_rewards(self):
        penalty = 0
        rewards = 0

        # number of jobs in backlog
        penalty += len(self.backlog) * self.returns["backlog_penalty"]

        # number of jobs remaining to schedule
        for job in self.state["job"]:
            if job.image != "":
                penalty += self.returns["job_remain_penalty"]
        
        # peak efficiency returns
        # pe = self.get_peak_efficiency(m)
        # if pe in range(self.pe_threshold):
        #     reward += returns["peak_eff_reward"]
        # else:
        #     penalty += returns["peak_eff_penalty"]
            
        total_return = reward + penalty

    def _step(self, action):
        if action not in range(self.job_queue_size + 1):
            raise ValueError("invalid job index") 

        MoveOn = False # Flag to allow multiple actions in one timestep
        if action == self.job_queue_size:
            MoveOn = True
        elif self.job_map[action] == False:
            MoveOn = True
        else:
            allocd = self.start_job(action)

        # Invalid action, or failed to start a job
        if MoveOn == True or allocd == False:

            # No jobs running, backlog, or queue
            if  len(self.backlog) == 0 and \
                all(job.image is "" for job in self.state["jobs"]) and \
                len(self.machine.running_jobs) == 0:
                self._episode_ended = True

            # no rewards until invalid action, i.e. MoveOn == True
            self.get_rewards()

        if allocd and len(self.backlog) > 0:
                self.state["job"][action] = self.backlog.pop()

        self.observe()

        if self._episode_ended:
            return ts.termination(self._state, total_return)
        else:
            return ts.transition(self._state, total_return)

    def start_job(self, action):
        job = self.state["jobs"][action]
        resp = requests.post(address+'/schedule', json={"job": job, "worker": self.machine.name})
        if resp.status_code != 200:
            print("failed to start job, ", job)
            return False
        print(f'{job.image} sent to {self.machine.name}')
        self.state["jobs"][action] = Job()
        return True
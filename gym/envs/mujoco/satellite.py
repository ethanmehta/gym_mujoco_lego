import numpy as np
import quaternion


from gym import utils
from gym.envs.mujoco import mujoco_env

class SatelliteEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i = 0
        self.w_d = np.zeros(3)
        self.q_d = np.asarray([0.707, 0, 0.707, 0])
        self.total_reward = 0


        utils.EzPickle.__init__(self)
        FILE_PATH = 'satellite.xml' # Absolute path to your .xml MuJoCo scene file.
        # For instance if I had the file in a subfolder of the project where I
        # defined this custom environment I could say 
        # FILE_PATH = os.getcwd() + '/custom_envs/assets/simple_env.xml'
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, 5)

        self.w_init = np.linalg.norm(self._get_obs()[1] - self.w_d)
        
        q_init_e = quaternion.as_float_array(quaternion.from_float_array(self._get_obs()[0])*quaternion.from_float_array(self.q_d))
        self.q_init = np.abs(q_init_e@np.asarray([0, 0, 0, 1]).T) - 1


    def flatten_observation(self, observation):
        return np.concatenate([obs.flatten() for obs in observation])

    def step(self, a):
        # Carry out one step 
        # Don't forget to do self.do_simulation(a, self.frame_skip)
        
        self.do_simulation(a, self.frame_skip)
        # print(self._get_obs())
        self.i += 1
        
        
        observation = self._get_obs()
        reward, done = self.reward(observation, a)

        self.total_reward += reward
        performance = (1200/(1-(self.total_reward/self.i)))*100
        print(performance)
        
        return self.flatten_observation(observation), reward, done, {"Count": self.i}
        
    def reward(self, state, action):
        alpha_q = 0.5
        alpha_w = 0.5
        
        # quaterion object
        q_e = quaternion.as_float_array(quaternion.from_float_array(state[0])*quaternion.from_float_array(self.q_d))

        w_e = state[1] - self.w_d

        q_err = np.abs(q_e@np.asarray([0, 0, 0, 1]).T) - 1

        c = 0
        done = False
        q_eps = 0.100
        w_eps = 0.100

        if (q_err <= q_eps):
            if (np.linalg.norm(w_e) <= w_eps):
                c = 1000
            else:
                c = 200
        elif (q_err >= 2*self.q_init) or np.linalg.norm(w_e) <= 2*self.w_init:
            done = True
            c = -10**4
        elif (q_err >= self.q_init) or np.linalg.norm(w_e) <= self.w_init:
            c = -10**3
        else:
            c = 0

        if (self.i > 900):
            done = True

        reward = -alpha_q*q_err - alpha_w*np.linalg.norm(w_e) - np.square(action).sum() - c

        return reward, done


    def viewer_setup(self):
        # Position the camera
        pass

    def reset_model(self):
        # Reset model to original state. 
        # This is called in the overall env.reset method
        # do not call this method directly. 
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        return self.flatten_observation(self._get_obs())

    def _get_obs(self):
      # Observation of environment feed to agent. This should never be called
      # directly but should be returned through reset_model and step
        return np.asarray([
            self.sim.data.get_body_xquat("satellite"),
            self.sim.data.get_body_xvelr("satellite"),
            self.sim.data.get_joint_qvel("rw_roll"),
            self.sim.data.get_joint_qvel("rw_pitch"),
            self.sim.data.get_joint_qvel("rw_yaw")
        ], dtype="object")
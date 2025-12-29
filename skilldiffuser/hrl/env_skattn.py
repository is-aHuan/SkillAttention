import numpy as np
import copy
import gym
from PIL import Image

from utils import String, lorl_gt_reward, lorl_save_im


class LorlWrapper(gym.Wrapper):
    """BabyAI processing wrapper.

        1) Adds String command to observations
        2) Preprocess states
    """

    def __init__(self, env, dataset, **kwargs):
        super(LorlWrapper, self).__init__(env)
        self.env = env
        self.dataset = dataset

        # ----- 两路归一化参数 -----
        # dataset 里最好存 torch.Tensor，这里统一转成 numpy
        self.proprio_scale  = np.ones((15,)).astype(np.float32)  # (Dp,)
        self.proprio_offset = np.zeros((15,)).astype(np.float32)  #

        # self.image_scale  = np.asarray(getattr(dataset, "image_scale")).astype(np.float32)   # (C,)
        # self.image_offset = np.asarray(getattr(dataset, "image_offset")).astype(np.float32)  # (C,)
        self.image_scale  = np.ones((3,)).astype(np.float32)  # (C,)
        self.image_offset = np.zeros((3,)).astype(np.float32)  # (C,)

        self.initial_state = None
        self.instr = kwargs["instr"]
        self.orig_instr = kwargs["orig_instr"]

    def reset(self, render=False, **kwargs):
        if render:
            render_path, iter_num, i = kwargs['render_path'], kwargs['iter_num'], kwargs["i"]

        env = self.env
        orig_instr, instr = self.orig_instr, self.instr
        im, _ = env.reset()

        # Initialize state for different tasks
        if orig_instr == "open drawer":
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
        elif orig_instr == "close drawer":
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == "turn faucet right":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == "turn faucet left":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == "move black mug right":
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == "move white mug down":
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        # Dont know if the following are correct
        elif orig_instr == 'open drawer and move black mug right':
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'pull the handle and move black mug down':
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move white mug right':
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move black mug down':
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'close drawer and turn faucet right':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == 'close drawer and turn faucet left':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == 'turn faucet left and move white mug down':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'turn faucet right and close drawer':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move white mug down and turn faucet left':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'close the drawer, turn the faucet left and move black mug right':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif instr == "open drawer and turn faucet counterclockwise":
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif instr == "slide the drawer closed and then shift white mug down":
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)

        # if orig_instr == "move white mug down":
        #    env._reset_hand(pos=[-0.1, 0.55, 0.1])
        # elif orig_instr == "move black mug right":
        #    env._reset_hand(pos=[-0.1, 0.55, 0.1])
        if "mug" in orig_instr:
            env._reset_hand(pos=[-0.1, 0.55, 0.1])
        else:
            env._reset_hand(pos=[0, 0.45, 0.1])

        for _ in range(50):
            env.sim.step()

        reset_state = copy.deepcopy(env.sim.data.qpos[:])
        env.sim.data.qpos[:] = reset_state
        env.sim.data.qacc[:] = 0
        env.sim.data.qvel[:] = 0
        env.sim.step()
        self.initial_state = copy.deepcopy(env.sim.data.qpos[:])

        if render:
            # Initialize goal image for initial state
            if orig_instr == "open drawer":
                env.sim.data.qpos[14] = -0.15
            elif orig_instr == "close drawer":
                env.sim.data.qpos[14] = 0.0
            elif orig_instr == "turn faucet right":
                env.sim.data.qpos[13] -= np.pi/5
            elif orig_instr == "turn faucet left":
                env.sim.data.qpos[13] += np.pi/5
            elif orig_instr == "move black mug right":
                env.sim.data.qpos[11] -= 0.1
            elif orig_instr == "move white mug down":
                env.sim.data.qpos[10] += 0.1

            env.sim.step()
            gim = env._get_obs()[:, :, :3]

            # Reset inital state
            env.sim.data.qpos[:] = reset_state
            env.sim.data.qacc[:] = 0
            env.sim.data.qvel[:] = 0
            env.sim.step()

            im = env._get_obs()[:, :, :3]
            initim = im
            render_path = "./render_images"
            lorl_save_im(
                (initim * 255.0).astype(np.uint8),
                render_path + f"/initialim_{iter_num}_{i}_{instr}.jpg")
            lorl_save_im((gim*255.0).astype(np.uint8), render_path+f"/gim_{iter_num}_{i}_{instr}.jpg")

        im = env._get_obs()[:, :, :3]
        return self._make_obs(im)

    def step(self, action):
        im, _, _, info = self.env.step(action)
        dist, s = lorl_gt_reward(self.env.sim.data.qpos[:], self.initial_state, self.orig_instr)

        reward = 0
        success = 0
        if s:
            success = 1
            reward = dist

        info.update({'success': success})

        # 确保取到的是当前帧
        if hasattr(self.env, "_get_obs"):
            im = self.env._get_obs()[:, :, :3]
        else:
            im = im[:, :, :3]

        return self._make_obs(im), reward, s, info

    def get_state(self, obs):
        """Returns the observation and lang_cmd"""
        return self._make_obs(obs)

    def _make_obs(self, im_hwc):
        """
        im_hwc: (H,W,C) float in [0,1] or uint8 (都行，下面会转 float32)
        return: dict(proprio, image, lang)
        """
        # ----- proprio -----
        proprio = np.asarray(self.env.sim.data.qpos[:], dtype=np.float32)  # (Dp,)
        proprio = proprio * self.proprio_scale + self.proprio_offset

        # ----- image -----
        if im_hwc.dtype != np.float32:
            im_hwc = im_hwc.astype(np.float32)

        if im_hwc.max() > 1.0:
            im_hwc = im_hwc / 255.0

        im_chw = np.moveaxis(im_hwc[:, :, :3], 2, 0).astype(np.float32)  # (C,H,W)
        # per-channel scale/offset: (C,)
        im_chw = im_chw * self.image_scale[:, None, None] + self.image_offset[:, None, None]

        return {"proprio": proprio, "image": im_chw, "lang": self.instr}

    def get_image(self, h=1024, w=1024):
        obs_cam0 = self.sim.render(h, w, camera_name="cam0") / 255.
        obs_top = self.sim.render(h, w, camera_name="topview") / 255.
        obs = np.concatenate([obs_cam0, obs_top], axis=1)
        im = np.flip(obs, 0).copy()
        return (im[:, :, :3]*255.0).astype(np.uint8)


if __name__ == "__main__":
    import lorl_env
    import gym
    # from skilldiffuser.hrl.envs import LorlEnv
    env = gym.make("LorlEnv-v0")
    env = LorlWrapper(env, dataset=None, instr="open drawer", orig_instr="open drawer")
    obs = env.reset()
    print(obs)
    print(obs["proprio"].shape)
    print(obs["image"].shape)
    print("=" * 20)

    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs, reward, done, info)
    print(obs["proprio"].shape)
    print(obs["image"].shape)
    im = env.get_image()
    Image.fromarray(im).show()  # Show the image
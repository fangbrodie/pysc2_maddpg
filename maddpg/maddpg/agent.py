import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES

_PLAYER_RELATIVE = SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = SCREEN_FEATURES.unit_type.index
_SELECTED = SCREEN_FEATURES.selected.index
_UNIT_HIT_POINTS = SCREEN_FEATURES.unit_hit_points.index
_UNIT_ENERGY = SCREEN_FEATURES.unit_energy.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


def make_default_args(arg_names):
    default_args = []
    spatial_seen = False
    spatial_arguments = ["screen", "minimap", "screen2"]
    for k in arg_names:
        if k in spatial_arguments:
            spatial_seen = True
            continue
        else:
            assert not spatial_seen, "got %s argument after spatial argument" % k
            default_args.append([0])

    return tuple(default_args), spatial_seen

def convert_point_to_rectangle(point, delta, dim):
    def l(x):
        return max(0, min(x, dim - 1))

    p1 = [l(k - delta) for k in point]
    p2 = [l(k + delta) for k in point]
    return p1, p2

def arg_names():
    x = [[a.name for a in k.args] for k in actions.FUNCTIONS]
    assert all("minimap2" not in k for k in x)
    return x

def find_rect_function_id():
    """
    this is just a change-safe way to return 3
    """
    x = [k.id for k, names in zip(actions.FUNCTIONS, arg_names()) if "screen2" in names]
    assert len(x) == 1
    return x[0]

class MADDPG(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""

  def __init__(self):
    super(MADDPG, self).__init__()
    self.default_args, is_spatial = zip(*[make_default_args(k) for k in arg_names()])
    self.is_spatial = np.array(is_spatial)
    self.rect_select_action_id = find_rect_function_id()
    self.rect_delta = 5
    self.dim = 40

  def selected_units(self, obs):
    selected = obs.observation["screen"][_SELECTED]
    self.selected_units_y, self.selected_units_x = selected.nonzero()
    self.num_units = len(self.selected_units_y)

  def select_unit(self, i, obs, env):
    if _SELECT_CONTROL_GROUP in obs.observation["available_actions"]:
      action = [actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0], [i]])]
      obs = env.step(action)[0]
      if obs.last():
          return False
    return obs

  def build_group(self, obs, env):
    obs_shape_n = []
    group = []
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    army_y, army_x = (player_relative == _PLAYER_HOSTILE).nonzero()
    for i in range(self.num_units):
      if _SELECT_UNIT in obs.observation["available_actions"]:
        action = [actions.FunctionCall(_SELECT_UNIT, [[0], [i]])]
        obs = env.step(action)[0]
      if _SELECT_CONTROL_GROUP in obs.observation["available_actions"]:
        action = [actions.FunctionCall(_SELECT_CONTROL_GROUP, [[2], [i]])]
        obs = env.step(action)[0]
        group.append(True)
      if _SELECT_ARMY in obs.observation["available_actions"]:
        action = [actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])]
        obs = env.step(action)[0]
      obs_shape_n.append((20,))
    self.group = group
    if _SELECT_UNIT in obs.observation["available_actions"]:
        action = [actions.FunctionCall(_SELECT_UNIT, [[2], [0]])]
        obs = env.step(action)[0]
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
        index = np.argmax(army_y)
        target = [army_x[index], army_y[index]]
        action = [actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
    if len(group) < self.num_units:
        self.selected_units(obs)
        self.build_group(obs, env)
    return obs_shape_n, obs

  def get_obs(self, obs, env):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    reaper_y, reaper_x =(player_relative == _PLAYER_FRIENDLY).nonzero()
    if len(reaper_x) == 5 and self.group.count(True) == 0:
        self.num_units = 5
        obs_shape_n, obs = self.build_group(obs, env)
    unit_y, unit_x = player_relative.nonzero()
    num_units = len(unit_x)

    obs_n = []
    unit = 0

    for i, alive_reaper in enumerate(self.group):
      obs_info = []
      if alive_reaper and _SELECT_CONTROL_GROUP in obs.observation["available_actions"]:
        action = [actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0], [unit]])]
        obs = env.step(action)[0]
        unit += 1
        self.selected_units(obs)
        if self.num_units == 1:
          for i in range(min(num_units, 10)):
            obs_info.append(unit_y[i] - self.selected_units_y[0])
            obs_info.append(unit_x[i] - self.selected_units_x[0])
          for i in range(num_units, 10):
              obs_info.append(0)
              obs_info.append(0)
        else:
            self.group[i] = False
            obs_info = [0] * 20
      elif not alive_reaper:
          obs_info = [0] * 20
      obs_n.append(obs_info)
    return obs_n, obs

  def make_one_action(self, action_id, spatial_coordinates):
    args = list(self.default_args[action_id])
    assert all(s < self.dim for s in spatial_coordinates)
    if action_id == self.rect_select_action_id:
      args.extend(convert_point_to_rectangle(spatial_coordinates, self.rect_delta, self.dim))
    elif self.is_spatial[action_id]:
      # NOTE: in pysc2 v 1.2 the action space (x,y) is flipped. Handling that conversion here
      # in all other places we operate with the "non-flipped" coordinates
      args.append(spatial_coordinates[::-1])

    return actions.FunctionCall(action_id, args)

  def step(self, obs):
    super(MADDPG, self).step(obs)
    if self.action_id in obs.observation["available_actions"]:
      return actions.FunctionCall(self.action_id, self.args)

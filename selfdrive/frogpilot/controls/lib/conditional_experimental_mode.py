from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import interp
from openpilot.common.params import Params

from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_functions import MovingAverageCalculator
from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_variables import CITY_SPEED_LIMIT, CRUISING_SPEED, PROBABILITY

class ConditionalExperimentalMode:
  def __init__(self):
    self.params_memory = Params("/dev/shm/params")

    self.experimental_mode = False
    self.lead_detected = False

    self.previous_status_value = 0
    self.status_value = 0

    self.lead_detection_mac = MovingAverageCalculator()

  def update(self, carState, enabled, frogpilotNavigation, modelData, radarState, road_curvature, t_follow, v_ego, frogpilot_toggles):
    lead = radarState.leadOne
    lead_distance = lead.dRel
    standstill = carState.standstill
    v_lead = lead.vLead

    self.update_conditions(lead_distance, lead.status, modelData, road_curvature, standstill, t_follow, v_ego, v_lead, frogpilot_toggles)

    condition_met = self.check_conditions(carState, frogpilotNavigation, lead, modelData, standstill, v_ego, frogpilot_toggles) and enabled
    if condition_met:
      self.experimental_mode = True
    else:
      self.experimental_mode = False
      self.status_value = 0

    if self.status_value != self.previous_status_value:
      self.params_memory.put_int("CEStatus", self.status_value)
      self.previous_status_value = self.status_value

  def check_conditions(self, carState, frogpilotNavigation, lead, modelData, standstill, v_ego, frogpilot_toggles):
    if standstill:
      self.status_value = 0
      return self.experimental_mode

    if (not self.lead_detected and v_ego <= frogpilot_toggles.conditional_limit) or (self.lead_detected and v_ego <= frogpilot_toggles.conditional_limit_lead):
      self.status_value = 10 if self.lead_detected else 11
      return True

    return False

  def update_conditions(self, lead_distance, lead_status, modelData, road_curvature, standstill, t_follow, v_ego, v_lead, frogpilot_toggles):
    self.lead_detection(lead_status)

  def lead_detection(self, lead_status):
    self.lead_detection_mac.add_data(lead_status)
    self.lead_detected = self.lead_detection_mac.get_moving_average() >= PROBABILITY

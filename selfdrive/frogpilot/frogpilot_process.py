import datetime
import http.client
import os
import socket
import time
import urllib.error
import urllib.request

import cereal.messaging as messaging

from cereal import car, log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.time import system_time_valid
from openpilot.system.hardware import HARDWARE

from openpilot.selfdrive.frogpilot.controls.frogpilot_planner import FrogPilotPlanner
from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_functions import FrogPilotFunctions
from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_variables import FrogPilotVariables
from openpilot.selfdrive.frogpilot.controls.lib.model_manager import DEFAULT_MODEL, DEFAULT_MODEL_NAME, download_model, populate_models
from openpilot.selfdrive.frogpilot.controls.lib.theme_manager import ThemeManager

WIFI = log.DeviceState.NetworkType.wifi

def github_pinged(url="https://github.com", timeout=5):
  try:
    urllib.request.urlopen(url, timeout=timeout)
    return True
  except (urllib.error.URLError, socket.timeout, http.client.RemoteDisconnected):
    return False

def automatic_update_check(params):
  update_available = params.get_bool("UpdaterFetchAvailable")
  update_ready = params.get_bool("UpdateAvailable")
  update_state = params.get("UpdaterState", encoding='utf8')

  if update_ready:
    HARDWARE.reboot()
  elif update_available:
    os.system("pkill -SIGHUP -f selfdrive.updated.updated")
  elif update_state == "idle":
    os.system("pkill -SIGUSR1 -f selfdrive.updated.updated")

def time_checks(automatic_updates, deviceState, params, params_memory):
  if github_pinged():
    populate_models()

    screen_off = deviceState.screenBrightnessPercent == 0
    wifi_connection = deviceState.networkType == WIFI

    if screen_off and wifi_connection:
      if automatic_updates:
        automatic_update_check(params)

      update_maps(params, params_memory)

def update_maps(params, params_memory):
  now = datetime.datetime.now()

  is_first = now.day == 1
  is_Sunday = now.weekday() == 6
  maps_selected = params.get("MapsSelected")
  schedule = params.get_int("PreferredSchedule")

  if maps_selected is None or schedule == 0 or (schedule == 1 and not is_Sunday) or (schedule == 2 and not is_first):
    return

  day = now.day
  suffix = "th" if 4 <= day <= 20 or 24 <= day <= 30 else ["st", "nd", "rd"][day % 10 - 1]
  todays_date = now.strftime(f"%B {day}{suffix}, %Y")

  if params.get("LastMapsUpdate") == todays_date:
    return

  if params.get("OSMDownloadProgress") is None:
    params_memory.put("OSMDownloadLocations", params.get("MapsSelected"))
    params.put("LastMapsUpdate", todays_date)

def frogpilot_thread(frogpilot_toggles):
  config_realtime_process(5, Priority.CTRL_LOW)

  params = Params()
  params_memory = Params("/dev/shm/params")

  frogpilot_functions = FrogPilotFunctions()
  theme_manager = ThemeManager()

  CP = None

  first_run = True
  model_list_empty = params.get("AvailableModelsNames", encoding='utf-8') is None
  time_validated = system_time_valid()

  pm = messaging.PubMaster(['frogpilotPlan'])
  sm = messaging.SubMaster(['carState', 'controlsState', 'deviceState', 'frogpilotCarControl', 'frogpilotCarState', 'frogpilotNavigation',
                            'frogpilotPlan', 'liveLocationKalman', 'longitudinalPlan', 'modelV2', 'radarState'],
                           poll='modelV2', ignore_avg_freq=['radarState'])

  while True:
    sm.update()

    deviceState = sm['deviceState']
    started = deviceState.started

    if started:
      if CP is None:
        with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
          CP = msg
          frogpilot_planner = FrogPilotPlanner(CP)

      if sm.updated['modelV2']:
        frogpilot_planner.update(sm['carState'], sm['controlsState'], sm['frogpilotCarControl'], sm['frogpilotCarState'],
                                 sm['frogpilotNavigation'], sm['liveLocationKalman'], sm['modelV2'], sm['radarState'], frogpilot_toggles)
        frogpilot_planner.publish(sm, pm, frogpilot_toggles)

    if params_memory.get("ModelToDownload", encoding='utf-8') is not None:
      download_model()

    if FrogPilotVariables.toggles_updated:
      FrogPilotVariables.update_frogpilot_params(started)
      frogpilot_toggles = FrogPilotVariables.toggles

      if not frogpilot_toggles.model_selector:
        params.put("Model", DEFAULT_MODEL)
        params.put("ModelName", DEFAULT_MODEL_NAME)

      if not started:
        frogpilot_functions.backup_toggles()

    if not time_validated:
      time_validated = system_time_valid()
      if not time_validated:
        continue

    if datetime.datetime.now().second == 0 or first_run or model_list_empty or params_memory.get_bool("ManualUpdateInitiated"):
      if not started or model_list_empty:
        time_checks(frogpilot_toggles.automatic_updates, deviceState, params, params_memory)
        model_list_empty = params.get("AvailableModelsNames", encoding='utf-8') is None

      theme_manager.update_holiday()

      first_run = False

    time.sleep(DT_MDL)

def main():
  frogpilot_thread(FrogPilotVariables.toggles)

if __name__ == "__main__":
  main()

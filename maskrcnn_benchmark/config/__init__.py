# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .defaults import _C

cfg = _C
def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

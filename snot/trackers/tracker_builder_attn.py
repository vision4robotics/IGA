from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from snot.core.config_attn import cfg
from snot.trackers.siamattn_tracker import SiamRPNTracker
from snot.trackers.siamattn_tracker import SiamMaskTracker
from snot.trackers.siamattn_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker_attn(model):
    return TRACKS[cfg.TRACK.TYPE](model)

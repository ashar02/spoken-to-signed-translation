
from pose_format import Pose

from ..text_to_gloss.types import Gloss
from .concatenate import concatenate_poses
from .lookup import PoseLookup, CSVPoseLookup
from typing import Union

pose_lookup_g = CSVPoseLookup('assets/dummy_lexicon')

def gloss_to_pose(glosses: Gloss, pose_lookup: Union[PoseLookup, None], spoken_language: str, signed_language: str, source: str = None) -> Pose:
    global pose_lookup_g
    if pose_lookup is None:
        if pose_lookup_g is None:
            pose_lookup_g = CSVPoseLookup('assets/dummy_lexicon')
        else:
            pose_lookup = pose_lookup_g
    # Transform the list of glosses into a list of poses
    poses = pose_lookup.lookup_sequence_db(glosses, spoken_language, signed_language, source)

    # Concatenate the poses to create a single pose
    return concatenate_poses(poses)

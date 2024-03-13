from typing import Union

from pose_format import Pose

from .concatenate import concatenate_poses
from .lookup import PoseLookup, CSVPoseLookup
from ..text_to_gloss.types import Gloss

pose_lookup_g = CSVPoseLookup('assets/dummy_lexicon')

def gloss_to_pose(glosses: Gloss,
                  pose_lookup: Union[PoseLookup, None],
                  spoken_language: str,
                  signed_language: str,
                  source: str = None,
                  anonymize: Union[bool, Pose] = False) -> Pose:
    global pose_lookup_g
    if pose_lookup is None:
        if pose_lookup_g is None:
            pose_lookup_g = CSVPoseLookup('assets/dummy_lexicon')
        else:
            pose_lookup = pose_lookup_g

    # Transform the list of glosses into a list of poses
    poses = pose_lookup.lookup_sequence_db(glosses, spoken_language, signed_language, source)

    # Anonymize poses
    if anonymize:
        try:
            from pose_anonymization.appearance import remove_appearance, transfer_appearance
        except ImportError:
            raise ImportError("Please install pose_anonymization. "
                              "pip install git+https://github.com/sign-language-processing/pose-anonymization")

        if isinstance(anonymize, Pose):
            print("Transferring appearance...")
            poses = [transfer_appearance(pose, anonymize) for pose in poses]
        else:
            print("Removing appearance...")
            poses = [remove_appearance(pose) for pose in poses]

    # Concatenate the poses to create a single pose
    return concatenate_poses(poses)

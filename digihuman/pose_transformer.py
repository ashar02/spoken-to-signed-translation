import numpy as np
from .pose_estimator import add_extra_points
#from pose_format import Pose


def Complete_pose_Buffer(pose):
    frame_count = pose.body.data.shape[0]
    # body_data = pose.body.data[frame_number]  # Get data for the specific frame
    rows, cols = pose.header.dimensions.height, pose.header.dimensions.width

    pose_landmarks_data = pose.get_components(["POSE_LANDMARKS"]).body
    lh_landmarks_data = pose.get_components(["LEFT_HAND_LANDMARKS"]).body
    rh_landmarks_data = pose.get_components(["RIGHT_HAND_LANDMARKS"]).body

    frameArr = []  # final response
    bodyPoseArr = []
    lhPoseArr = []
    rhPoseArr = []

    # Loop to populate bodyPoseArr
    for frame_index, (body_frame_data, body_conf) in enumerate(
            zip(pose_landmarks_data.data, pose_landmarks_data.confidence)):
        body_predictions = []
        for landmark_index, (body_landmark, body_conf_value) in enumerate(zip(body_frame_data[0], body_conf[0])):
            body_pose_pred = {
                'x': (body_landmark[0] / rows),
                'y': (body_landmark[1] / cols),
                'z': ((body_landmark[2] / 500) - 1.25),
                'visibility': body_conf_value
            }
            if isinstance(body_landmark[0], np.float32):
                body_predictions.append(body_pose_pred)
        add_extra_points(body_predictions)
        body_pose_obj = {
            "predictions": body_predictions,
            "frame": frame_index + 1,
            "height": rows,
            "width": cols
        }
        bodyPoseArr.append(body_pose_obj)

    # Loop to populate lhPoseArr
    for frame_index, (lh_frame_data, lh_conf) in enumerate(
            zip(lh_landmarks_data.data, lh_landmarks_data.confidence)):
        lh_predictions = []
        for landmark_index, (lh_landmark, lh_conf_value) in enumerate(zip(lh_frame_data[0], lh_conf[0])):
            lh_pose_pred = {
                'x': (lh_landmark[0] / rows),
                'y': (lh_landmark[1] / cols),
                'z': ((lh_landmark[2] / 500) - 1.25),
                'visibility': (lh_conf_value)
            }
            if isinstance(lh_landmark[0], np.float32):
                lh_predictions.append(lh_pose_pred)
        lhPoseArr.append(lh_predictions)

    # Loop to populate rhPoseArr
    for frame_index, (rh_frame_data, rh_conf) in enumerate(
            zip(rh_landmarks_data.data, rh_landmarks_data.confidence)):
        rh_predictions = []
        for landmark_index, (rh_landmark, rh_conf_value) in enumerate(zip(rh_frame_data[0], rh_conf[0])):
            rh_pose_pred = {
                'x': (rh_landmark[0] / rows),
                'y': (rh_landmark[1] / cols),
                'z': ((rh_landmark[2] / 500) - 1.25),
                'visibility': rh_conf_value
            }
            if isinstance(rh_landmark[0], np.float32):
                rh_predictions.append(rh_pose_pred)
        rhPoseArr.append(rh_predictions)

    for i in range(0, frame_count):
        frame_object = {
            "bodyPose": {
                "predictions": bodyPoseArr[i]['predictions'],
                "frame": i + 1,
                "height": rows,
                "width": cols
            },
            "handsPose": {
                "handsR": rhPoseArr[i],
                "handsL": lhPoseArr[i],
                "frame": i + 1
            },
            "frame": i + 1
        }
        frameArr.append(frame_object)

    return frameArr


# Function to convert NumPy types to native Python types
def convert_numpy_to_native(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data

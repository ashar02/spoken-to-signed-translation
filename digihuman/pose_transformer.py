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
        body_predictions = Create_Dummy_Pose_Buffer()
        for landmark_index, (body_landmark, body_conf_value) in enumerate(zip(body_frame_data[0], body_conf[0])):
            body_pose_pred = {
                'x': float(body_landmark[0] / rows),
                'y': float(body_landmark[1] / cols),
                'z': float((body_landmark[2] / 500) - 1.25),
                'visibility': float(body_conf_value)
            }
            # if isinstance(body_landmark[0], np.float32):
            if landmark_index == 0:
                body_predictions[11] = body_pose_pred
            elif landmark_index == 1:
                body_predictions[12] = body_pose_pred
            elif landmark_index == 2:
                body_predictions[13] = body_pose_pred
            elif landmark_index == 3:
                body_predictions[14] = body_pose_pred
            elif landmark_index == 4:
                body_predictions[15] = body_pose_pred
            elif landmark_index == 5:
                body_predictions[16] = body_pose_pred
            elif landmark_index == 6:
                body_predictions[23] = body_pose_pred
            elif landmark_index == 7:
                body_predictions[24] = body_pose_pred
                # body_predictions.append(body_pose_pred)
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
                'x': float(lh_landmark[0] / rows),
                'y': float(lh_landmark[1] / cols),
                'z': float((lh_landmark[2] / 500) - 1.25),
                'visibility': float(lh_conf_value)
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
                'x': float(rh_landmark[0] / rows),
                'y': float(rh_landmark[1] / cols),
                'z': float((rh_landmark[2] / 500) - 1.25),
                'visibility': float(rh_conf_value)
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

def Create_Dummy_Pose_Buffer():
    body_pose_arr = []
    nose_point = {
        "x": 0.002864239737391472,
        "y": -0.002864239737391472,
        "z": -0.3728623688220978,
        "visibility": 0.9998102784156799
    }
    body_pose_arr.append(nose_point)
    left_eye_inner = {
        "x": 0.02357611432671547,
        "y": -0.5890240669250488,
        "z": -0.366853803396225,
        "visibility": 0.999498724937439
    }
    body_pose_arr.append(left_eye_inner)
    left_eye = {
        "x": 0.0257857758551836,
        "y": -0.5894666910171509,
        "z": -0.35470613837242126,
        "visibility": 0.9994004964828491
    }
    body_pose_arr.append(left_eye)
    left_eye_outer = {
            "x": 0.023005325347185135,
            "y": -0.5897130966186523,
            "z": -0.35937708616256714,
            "visibility": 0.9995629191398621
    }
    body_pose_arr.append(left_eye_outer)
    right_eye_inner = {
            "x": 0.0006337692029774189,
            "y": -0.5990504026412964,
            "z": -0.3776598870754242,
            "visibility": 0.9996899366378784
          }
    body_pose_arr.append(right_eye_inner)
    right_eye = {
            "x": 0.003401678055524826,
            "y": -0.5968003273010254,
            "z": -0.391887366771698,
            "visibility": 0.9997161030769348
          }
    body_pose_arr.append(right_eye)
    right_eye_outer = {
            "x": 0.005837739445269108,
            "y": -0.5848087668418884,
            "z": -0.3737899959087372,
            "visibility": 0.9997919201850891
          }
    body_pose_arr.append(right_eye_outer)
    left_ear = {
            "x": 0.08456195890903473,
            "y": -0.5845540761947632,
            "z": -0.26897335052490234,
            "visibility": 0.999743640422821
          }
    body_pose_arr.append(left_ear)
    right_ear = {
            "x": -0.05707802623510361,
            "y": -0.5472390055656433,
            "z": -0.2679259777069092,
            "visibility": 0.9996002316474915
          }
    body_pose_arr.append(right_ear)
    left_mouth = {
            "x": 0.0416022464632988,
            "y": -0.5585125684738159,
            "z": -0.3240182399749756,
            "visibility": 0.9998192191123962
    }
    body_pose_arr.append(left_mouth)
    right_mouth = {
            "x": 0.003893398679792881,
            "y": -0.5258322358131409,
            "z": -0.3489793539047241,
            "visibility": 0.9998238682746887
    }
    body_pose_arr.append(right_mouth)
    left_shoulder = {
            "x": 0.18054859340190887,
            "y": -0.41907933354377747,
            "z": -0.14147968590259552,
            "visibility": 0.9998019337654114
    }
    body_pose_arr.append(left_shoulder)
    right_shoulder = {
            "x": -0.14075763523578644,
            "y": -0.4917465150356293,
            "z": -0.0903409868478775,
            "visibility": 0.9996144771575928
    }
    body_pose_arr.append(right_shoulder)
    left_elbow = {
            "x": 0.25610026717185974,
            "y": -0.24118980765342712,
            "z": -0.1646481305360794,
            "visibility": 0.7287042140960693
    }
    body_pose_arr.append(left_elbow)
    right_elbow = {
            "x": -0.2337333858013153,
            "y": -0.2318800836801529,
            "z": -0.10753609240055084,
            "visibility": 0.6463595628738403
    }
    body_pose_arr.append(right_elbow)
    left_wrist = {
            "x": 0.23519833385944366,
            "y": -0.03708893060684204,
            "z": -0.1313290148973465,
            "visibility": 0.11594510823488235
    }
    body_pose_arr.append(left_wrist)
    right_wrist = {
            "x": -0.25856801867485046,
            "y": -0.011624474078416824,
            "z": -0.1720554232597351,
            "visibility": 0.11254721134901047
    }
    body_pose_arr.append(right_wrist)
    left_pinky = {
            "x": 0.22218184173107147,
            "y": 0.009835867211222649,
            "z": -0.13051144778728485,
            "visibility": 0.10091392695903778
    }
    body_pose_arr.append(left_pinky)
    right_pinky = {
            "x": -0.23083502054214478,
            "y": 0.030628308653831482,
            "z": -0.1981191337108612,
            "visibility": 0.09208588302135468
    }
    body_pose_arr.append(right_pinky)
    left_index = {
            "x": 0.18733002245426178,
            "y": 0.004402919672429562,
            "z": -0.14337453246116638,
            "visibility": 0.13129669427871704
    }
    body_pose_arr.append(left_index)
    right_index = {
            "x": -0.19222745299339294,
            "y": 0.005089309066534042,
            "z": -0.22431889176368713,
            "visibility": 0.1254805624485016
    }
    body_pose_arr.append(right_index)
    left_thumb = {
            "x": 0.22298789024353027,
            "y": -0.012500501237809658,
            "z": -0.13968704640865326,
            "visibility": 0.12276943773031235
    }
    body_pose_arr.append(left_thumb)
    right_thumb = {
            "x": -0.23845353722572327,
            "y": -0.009463272988796234,
            "z": -0.18911387026309967,
            "visibility": 0.12625083327293396
    }
    body_pose_arr.append(right_thumb)
    left_hip = {
            "x": 0.09153936803340912,
            "y": 0.006380623206496239,
            "z": 0.028282305225729942,
            "visibility": 0.20512712001800537
          }
    body_pose_arr.append(left_hip)
    right_hip = {
            "x": -0.08790028095245361,
            "y": -0.012431442737579346,
            "z": -0.03012768179178238,
            "visibility": 0.2503540813922882
          }
    body_pose_arr.append(right_hip)
    left_knee = {
            "x": 0.06759689003229141,
            "y": 0.3989599645137787,
            "z": 0.004279731307178736,
            "visibility": 0.0016407257644459605
          }
    body_pose_arr.append(left_knee)
    right_knee = {
            "x": -0.02197316102683544,
            "y": 0.3432997763156891,
            "z": -0.06096237525343895,
            "visibility": 0.004357140976935625
          },
    body_pose_arr.append(right_knee)
    left_ankle = {
            "x": 0.09173575043678284,
            "y": 0.7143592238426208,
            "z": 0.16469673812389374,
            "visibility": 0.0010296484688296914
          }
    body_pose_arr.append(left_ankle)
    right_ankle = {
            "x": -0.07791144400835037,
            "y": 0.715643584728241,
            "z": 0.17496968805789948,
            "visibility": 0.0016159728402271867
          }
    body_pose_arr.append(right_ankle)
    left_heel = {
            "x": 0.07568726688623428,
            "y": 0.7432113885879517,
            "z": 0.18743547797203064,
            "visibility": 0.0014383119996637106
          }
    body_pose_arr.append(left_heel)
    right_heel = {
            "x": -0.07841628789901733,
            "y": 0.7465903759002686,
            "z": 0.16193915903568268,
            "visibility": 0.0014402027009055018
          }
    body_pose_arr.append(right_heel)
    left_foot_index = {
            "x": 0.07058921456336975,
            "y": 0.7296340465545654,
            "z": 0.13124611973762512,
            "visibility": 0.0012406683526933193
          }
    body_pose_arr.append(left_foot_index)
    right_foot_index = {
            "x": -0.06677509844303131,
            "y": 0.7724581956863403,
            "z": 0.10267645120620728,
            "visibility": 0.0014662168687209487
          }
    body_pose_arr.append(right_foot_index)
    return body_pose_arr
        #   {
        #     "x": 0.0018195435404777527,
        #     "y": -0.0030254097655415535,
        #     "z": -0.0009226882830262184,
        #     "visibility": 0.2277406007051468
        #   },
        #   {
        #     "x": 0.010857511311769485,
        #     "y": -0.22921916702762246,
        #     "z": -0.058416512329131365,
        #     "visibility": 0.6137244030833244
        #   },
        #   {
        #     "x": 0.02132165082730353,
        #     "y": -0.4987926632165909,
        #     "z": -0.22620456665754318,
        #     "visibility": 0.9997648745775223
        #   },
        #   {
        #     "x": 0.010116057470440865,
        #     "y": -0.5658950010935465,
        #     "z": -0.3032538990179698,
        #     "visibility": 0.9997180501619974
        #   }


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

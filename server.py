import os
from flask import Flask, request, jsonify, Response
from spoken_to_signed.text_to_gloss.spacylemma import text_to_gloss
from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup
from dotenv import load_dotenv
from io import BytesIO
from flask_compress import Compress
from gunicorn.app.base import BaseApplication
from pose_format.utils.holistic import load_holistic
from pose_format.utils.generic import reduce_holistic
import cv2
from datetime import datetime
from typing import Optional
from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
from typing import List
from pose_format import Pose
from spoken_to_signed.pose_to_video.conditional.pix2pix import pose_to_video_pix2pix
from tqdm import tqdm
import tempfile
import uuid
import mimetypes
from digihuman.pose_estimator import Complete_pose_Video
from digihuman.mediaPipeFace import Calculate_Face_Mocap
import ssl
import time
from pymongo import MongoClient


full_pose_video_data = {}
full_pose_video_data_statues = {}
face_pose_video_data = {}
face_pose_video_data_statues = {}
load_dotenv()
mimetypes.init()
TEMP_FILE_FOLDER = "temp/"
app = Flask(__name__)
max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', 10)) * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = max_content_length
Compress(app)

connection_string = os.getenv('DB_URI')
client = MongoClient(connection_string)
db = client['translate']

def remove_unsupported_characters(text):
    return ''.join(char for char in text if ord(char) <= 255)

@app.route('/spoken_text_to_signed_pose', methods=['GET'])
def text_to_posses():
    worker_pid = os.getpid()
    print(f"Worker PID: {worker_pid}")
    try:
        text = request.args.get('text').lower()
        language = request.args.get('spoken')
        signed = request.args.get('signed')
        mode = request.args.get('mode')
        if not text or not language or not signed:
            return jsonify({'error': 'Missing required fields: text or spoken or signed'}), 400
        print('request: ' + text)

        glosses = text_to_gloss(text, language)
        poses = gloss_to_pose(glosses, None, language, signed)
        extension = '.pose'
        content_type = 'application/pose'
        if mode and mode == "2":
            extension = '.mp4'
            content_type = 'video/mp4'
        #with open('final.pose', "wb") as f:
        #    poses.write(f)

        gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
        supported_words = '-'.join([word for word, _ in glosses])
        filename = supported_words + extension
        headers = {
            #'Cache-Control': 'public, max-age=3600',
            'Cache-Control': 'no-store',
            'Content-Disposition': f'inline; filename="{filename}"',
            'Content-Type': content_type,
            'Glosses': gloss_sequence
        }
        #response = jsonify({'glosses': glosses})
        binary_data = None
        if mode and mode == "2":
            print('Generating video ...')
            video = None
            frames: iter = pose_to_video_pix2pix(poses)
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            for frame in tqdm(frames):
                if video is None:
                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    video = cv2.VideoWriter(temp_file.name,
                                            apiPreference=cv2.CAP_FFMPEG,
                                            fourcc=fourcc,
                                            fps=poses.body.fps,
                                            frameSize=(width, height))
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)
            video.release()
            with open(temp_file.name, 'rb') as f:
                binary_data = f.read()
            #with open('./test.mp4', 'rb') as f:
            #    binary_data = f.read()
            os.remove(temp_file.name)
        else:
            buffer = BytesIO()
            poses.write(buffer)
            binary_data = buffer.getvalue()
            buffer.close()
        response = Response(binary_data, mimetype=content_type)
        for key, value in headers.items():
            cleaned_key = remove_unsupported_characters(key)
            cleaned_value = remove_unsupported_characters(value)
            response.headers[cleaned_key] = cleaned_value
        return response
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_message = str(e)
        print(f"Unexpected error: {error_message}")
        if os.getenv('PM2_HOME'):
            return jsonify(error_message), 500
        else:
            return jsonify({'message': error_message}), 500

@app.route('/video_to_pose', methods=['POST'])
def video_to_pose():
    worker_pid = os.getpid()
    print(f"Worker PID: {worker_pid}")
    try:
        word = request.form.get('word', '').lower()
        language = request.form.get('spoken')
        signed = request.form.get('signed')
        gloss_sequence = None
        if word and language and signed:
            print('request: ' + word)
            glosses = text_to_gloss(word, language)
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])

        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify('File size exceeds maximum limit'), 413
        if 'video' not in request.files:
            return jsonify('No video file found'), 400
        video_files = request.files.getlist('video')
        if len(video_files) != 1:
            return jsonify('Only one video file is allowed'), 400
        video_file = video_files[0]
        if video_file.filename == '':
            return jsonify('Empty video filename'), 400
        allowed_extensions = {'mp4', 'webm'}
        if '.' not in video_file.filename or video_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify('Invalid video file format, must be mp4 or webm'), 400

        video_pose_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_pose')
        if not os.path.exists(video_pose_directory):
            os.makedirs(video_pose_directory)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename, file_extension = os.path.splitext(video_file.filename)
        unique_filename = f"{filename}_{timestamp}{file_extension}"
        video_path = os.path.join(video_pose_directory, unique_filename)
        video_file.save(video_path)
        # Call pose_video function from pose_estimation.py module
        pose_data = pose_video(video_path, None, 'mediapipe')
        poses: List[Pose] = []
        poses.append(pose_data)
        pose_data = concatenate_poses(poses)
        os.remove(video_path)
        headers = {
            #'Cache-Control': 'public, max-age=3600',
            'Cache-Control': 'no-store',
            'Content-Disposition': f'inline; filename="{filename}.pose"',
            'Content-Type': 'application/pose',
        }
        if gloss_sequence:
            headers['Glosses'] = gloss_sequence
        buffer = BytesIO()
        pose_data.write(buffer)
        binary_data = buffer.getvalue()
        buffer.close()
        response = Response(binary_data, mimetype='application/pose')
        for key, value in headers.items():
            cleaned_key = remove_unsupported_characters(key)
            cleaned_value = remove_unsupported_characters(value)
            response.headers[cleaned_key] = cleaned_value
        return response
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_message = str(e)
        print(f"Unexpected error: {error_message}")
        if os.getenv('PM2_HOME'):
            return jsonify(error_message), 500
        else:
            return jsonify({'message': error_message}), 500

@app.route('/holisticUploader', methods=['POST'])
def upload_holistic_video():
    if 'file' not in request.files:
        return "No file!"
    f = request.files['file']
    postfix = f.filename.split(".")[-1]
    file_name = TEMP_FILE_FOLDER + str(uuid.uuid4()) + "." + postfix
    f.save(file_name)
    # checking file type
    mimestart = mimetypes.guess_type(file_name)[0]
    if mimestart != None:
        mimestart = mimestart.split('/')[0]
        if mimestart in ['video']:
            if os.getenv('PM2_HOME'):
                db.full_pose_video_data.insert_one({
                    'file_name': file_name,
                    'status': 'processing',
                    'created_at': datetime.utcnow(),
                    'poses': []
                })
            else:
                full_pose_video_data[file_name] = []
                full_pose_video_data_statues[file_name] = False
                print("request type video")
            calculate_video_full_pose_estimation(file_name)
            cap = cv2.VideoCapture(file_name)
            tframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            aspectRatio = width / height
            cap.release()
            res = {
                'file' : file_name,
                'totalFrames': int(tframe),
                'aspectRatio': aspectRatio
            }
            os.remove(file_name)
            return jsonify(res)
        else:
            print("Wrong input!")
            os.remove(file_name)
            return "Not video!"
    else:
        os.remove(file_name)
        return "No mime found!"

@app.route('/faceUploader', methods=['POST'])
def upload_face_video():
    if 'file' not in request.files:
        return "No file!"
    f = request.files['file']
    postfix = f.filename.split(".")[-1]
    file_name = TEMP_FILE_FOLDER + str(uuid.uuid4()) + "." + postfix
    f.save(file_name)
    # checking file type
    mimestart = mimetypes.guess_type(file_name)[0]
    if mimestart != None:
        mimestart = mimestart.split('/')[0]
        if mimestart in ['video']:
            if os.getenv('PM2_HOME'):
                db.face_pose_video_data.insert_one({
                    'file_name': file_name,
                    'status': 'processing',
                    'created_at': datetime.utcnow(),
                    'poses': []
                })
            else:
                face_pose_video_data[file_name] = []
                face_pose_video_data_statues[file_name] = False
                print("request type video")
            calculate_video_mocap_estimation(file_name)
            cap = cv2.VideoCapture(file_name)
            tframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            aspectRatio = width / height
            cap.release()
            res = {
                'file': file_name,
                'totalFrames': int(tframe),
                'aspectRatio': aspectRatio
            }
            os.remove(file_name)
            return jsonify(res)
        else:
            print("Wrong input!")
            os.remove(file_name)
            return "Not video!"
    else:
        os.remove(file_name)
        return "No mime found!"

@app.route("/holistc", methods=["POST"])
def get_frame_full_pose():
    if not request.is_json:
        return "No json body!"
    request_json = request.get_json()
    if 'index' not in request_json:
        return "No index!"
    if 'fileName' not in request_json:
        return "No fileName!"
    index = request_json['index']
    file_name = str(request_json['fileName'])
    req = request.data
    if os.getenv('PM2_HOME'):
        try:
            while True:
                video_data = db.full_pose_video_data.find_one({'file_name': file_name})
                if not video_data:
                    print("Wrong!")
                    return Response("Wrong fileName!")
                poses = video_data['poses']
                if len(poses) >= index + 1:
                    return jsonify(poses[index])
                elif video_data['status'] == 'processing':
                    time.sleep(0.1)
                else:
                    return Response("Done")
        except Exception as e:
            print(f"Error: {e}")
            return Response("Good luck!")
    else:
        try:
            if full_pose_video_data.keys().__contains__(file_name) is False:
                print("Wrong!")
                return Response("Wrong fileName!")
            while True:
                if len(full_pose_video_data[file_name]) >= index + 1:
                    # print(hand_pose_video_data[file_name][index])
                    return jsonify(full_pose_video_data[file_name][index])
                elif full_pose_video_data_statues[file_name] is False:
                    time.sleep(0.1)
                else:
                    return Response("Done")
        except:
            return Response("Good luck!")

@app.route("/face", methods=["POST"])
def get_frame_facial_expression():
    if not request.is_json:
        return "No json body!"
    request_json = request.get_json()
    if 'index' not in request_json:
        return "No index!"
    if 'fileName' not in request_json:
        return "No fileName!"
    index = request_json['index']
    file_name = str(request_json['fileName'])
    req = request.data
    if os.getenv('PM2_HOME'):
        try:
            while True:
                video_data = db.face_pose_video_data.find_one({'file_name': file_name})
                if not video_data:
                    print("Wrong!")
                    return Response("Wrong fileName!")
                poses = video_data['poses']
                if len(poses) >= index + 1:
                    return jsonify(poses[index])
                elif video_data['status'] == 'processing':
                    time.sleep(0.1)
                else:
                    return Response("Done")
        except Exception as e:
            print(f"Error: {e}")
            return Response("Good luck!")
    else:
        try:
            if face_pose_video_data.keys().__contains__(file_name) is False:
                print("Wrong!")
                return Response("Wrong fileName!")
            while True:
                if len(face_pose_video_data[file_name]) >= index + 1:
                    return jsonify(face_pose_video_data[file_name][index])
                elif face_pose_video_data_statues[file_name] is False:
                    time.sleep(0.1)
                else:
                    return Response("Done")
        except:
            return Response("Good luck!")

def load_video_frames(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def pose_video(input_path: str, output_path: Optional[str], format: str):
    # Load video frames
    print('Loading video ...')
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #frames = load_video_frames(cap)
    frames_generator = load_video_frames(cap)
    frames = list(frames_generator)
    cap.release()

    # Crop to square
    print('Cropping video to square ...')
    if width != height:
        crop_size = min(width, height)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        frames = [frame[top:top + crop_size, left:left + crop_size] for frame in frames]
        width = height = crop_size

    # Resize to 1250x1250
    target_size = 1250
    frames = [cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4) for frame in frames]
    width = height = target_size

    # Perform pose estimation
    print('Estimating pose ...')
    if format == 'mediapipe':
        pose = load_holistic(frames,
                             fps=fps,
                             width=width,
                             height=height,
                             progress=False,
                             additional_holistic_config={'model_complexity': 2}) #576 landmarks
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]) #543 landmarks
        pose = reduce_holistic(pose) #178 landmarks
    else:
        raise NotImplementedError('Pose format not supported')

    # Write
    if output_path:
        print('Saving to disk ...')
        with open(output_path, "wb") as f:
            pose.write(f)
    else:
        print('Returning pose data ...')
        return pose

def calculate_video_full_pose_estimation(file_name):
    if os.getenv('PM2_HOME'):
        video_data = db.full_pose_video_data.find_one({'file_name': file_name})
        if video_data:
            poses = video_data['poses']
            for index in Complete_pose_Video(file_name):
                poses.append(index)
            db.full_pose_video_data.update_one(
                {'file_name': file_name},
                {'$set': {'poses': poses, 'status': 'complete'}}
            )
        else:
            print(f"No document found for file_name: {file_name}")
    else:
        for i in Complete_pose_Video(file_name):
            full_pose_video_data[file_name].append(i)
        full_pose_video_data_statues[file_name] = True

def calculate_video_mocap_estimation(file_name):
    if os.getenv('PM2_HOME'):
        video_data = db.face_pose_video_data.find_one({'file_name': file_name})
        if video_data:
            poses = video_data['poses']
            for i in Calculate_Face_Mocap(file_name):
                poses.append(i)
            db.face_pose_video_data.update_one(
                {'file_name': file_name},
                {'$set': {'poses': poses, 'status': 'complete'}}
            )
        else:
            print(f"No document found for file_name: {file_name}")
    else:
        for i in Calculate_Face_Mocap(file_name):
            face_pose_video_data[file_name].append(i)
        face_pose_video_data_statues[file_name] = True


if __name__ == '__main__':
    isExist = os.path.exists(TEMP_FILE_FOLDER)
    if not isExist:
        os.makedirs(TEMP_FILE_FOLDER)
    ssl._create_default_https_context = ssl._create_unverified_context

    if os.getenv('PM2_HOME'):
        port_number = int(os.getenv('PORT', 3002))
        ssl_context = (os.getenv('CERTIFICATE_PATH'), os.getenv('PRIVATE_KEY_PATH'))
        options = {
            'bind': f'0.0.0.0:{port_number}',
            'workers': int(os.getenv('GUNICORN_WORKERS', 2)),
            'threads': int(os.getenv('GUNICORN_THREADS', 1)),
            'certfile': ssl_context[0],
            'keyfile': ssl_context[1],
            'loglevel': 'error'
        }
        class FlaskApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application
        FlaskApplication(app, options).run()
    else:
        port_number = int(os.getenv('PORT', 3002))
        ssl_context = (os.getenv('CERTIFICATE_PATH'), os.getenv('PRIVATE_KEY_PATH'))
        app.run(debug=True, host='0.0.0.0', port=port_number, ssl_context=ssl_context)
        print(f"Server listening on port: {port_number}")

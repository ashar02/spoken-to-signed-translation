import os
from flask import Flask, request, jsonify, Response
from spoken_to_signed.text_to_gloss.spacylemma import text_to_gloss
from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup
from dotenv import load_dotenv
from io import BytesIO
from flask_compress import Compress
from gunicorn.app.base import BaseApplication
from pose_format.utils.holistic import load_holistic
import cv2
from datetime import datetime
from typing import Optional

load_dotenv()
app = Flask(__name__)
max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', 10)) * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = max_content_length
Compress(app)

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
        if not text or not language or not signed:
            return jsonify({'error': 'Missing required fields: text or spoken or signed'}), 400
        print('request: ' + text)

        glosses = text_to_gloss(text, language)
        poses = gloss_to_pose(glosses, None, language, signed)
        #with open('final.pose', "wb") as f:
        #    poses.write(f)

        gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
        supported_words = '-'.join([word for word, _ in glosses])
        filename = supported_words + '.pose'
        headers = {
            #'Cache-Control': 'public, max-age=3600',
            'Cache-Control': 'no-store',
            'Content-Disposition': f'inline; filename="{filename}"',
            'Content-Type': 'application/pose',
            'Glosses': gloss_sequence
        }
        #response = jsonify({'glosses': glosses})
        buffer = BytesIO()
        poses.write(buffer)
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

@app.route('/video_to_pose', methods=['POST'])
def video_to_pose():
    worker_pid = os.getpid()
    print(f"Worker PID: {worker_pid}")
    try:
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
        os.remove(video_path)
        headers = {
            #'Cache-Control': 'public, max-age=3600',
            'Cache-Control': 'no-store',
            'Content-Type': 'application/pose',
        }
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

    # Perform pose estimation
    print('Estimating pose ...')
    if format == 'mediapipe':
        pose = load_holistic(frames,
                             fps=fps,
                             width=width,
                             height=height,
                             progress=True,
                             additional_holistic_config={'model_complexity': 1})
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

if __name__ == '__main__':
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

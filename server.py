import os
from flask import Flask, request, jsonify, Response
from spoken_to_signed.text_to_gloss.spacylemma import text_to_gloss
from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup
from dotenv import load_dotenv
from io import BytesIO
from flask_compress import Compress
from gunicorn.app.base import BaseApplication
import tempfile
# import subprocess
from pose_format.utils.holistic import load_holistic
import cv2
import shutil

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
Compress(app)

@app.route('/spoken_text_to_signed_pose', methods=['GET'])
def text_to_glosses():
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
            'Cache-Control': 'public, max-age=3600',
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
            response.headers[key] = value
        return response
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'message': str(e)}), 500

@app.route('/video_to_pose', methods=['POST'])
def video_to_pose():
    if 'video' not in request.files:
        return 'No video file found', 400

    video_file = request.files['video']

    if video_file.filename == '':
        return 'Empty video filename', 400

    if not allowed_file(video_file.filename):
        return 'Invalid video file format, must be mp4 or webm', 400
    
    if request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return 'File size exceeds maximum limit', 413

    # Create a temporary directory to store the uploaded video file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded video file to the temporary directory
    temp_video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(temp_video_path)

    # Specify the output pose file path
    output_pose_path = os.path.join(temp_dir, 'output.pose')

    # Call pose_video function from pose_estimation.py module
    pose_video(temp_video_path, output_pose_path, 'mediapipe')

    # Read the generated pose file
    with open(output_pose_path, 'rb') as f:
        pose_data = f.read()

    # Set response headers
    headers = {
        'Content-Disposition': f'inline; filename="output.pose"',
        'Content-Type': 'application/pose'
    }

    # Create a Flask response with the pose data
    response = Response(pose_data, headers=headers)

    # Remove the uploaded video file
    delete_file(temp_video_path)
    delete_file(output_pose_path)
    # Remove the temp directory of video file
    delete_temp_dir(temp_dir)

    return response

# Function to delete a file
def delete_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

# Function to delete a directory and its contents
def delete_temp_dir(temp_dir):
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error deleting temporary directory: {e}")

def load_video_frames(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def pose_video(input_path: str, output_path: str, format: str):
    # Load video frames
    print('Loading video ...')
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = load_video_frames(cap)

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
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        pose.write(f)


# Function to check if the file extension is allowed
def allowed_file(filename):
    allowed_extensions = {'mp4', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
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

import os
from flask import Flask, request, jsonify, Response
from spoken_to_signed.text_to_gloss.spacylemma import text_to_gloss
from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup
from dotenv import load_dotenv
from io import BytesIO
from flask_compress import Compress

load_dotenv()
app = Flask(__name__)
Compress(app)

@app.route('/spoken_text_to_signed_pose', methods=['GET'])
def text_to_glosses():
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

if __name__ == '__main__':
    port_number = int(os.getenv('PORT', 3002))
    ssl_context = (os.getenv('CERTIFICATE_PATH'), os.getenv('PRIVATE_KEY_PATH'))
    app.run(debug=True, host='0.0.0.0', port=port_number, ssl_context=ssl_context)
    print(f"Server listening on port: {port_number}")

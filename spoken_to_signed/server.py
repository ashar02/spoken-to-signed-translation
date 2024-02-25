from flask import Flask, request, jsonify
from text_to_gloss.spacylemma import text_to_gloss

app = Flask(__name__)

@app.route('/text-to-gloss', methods=['GET'])
def text_to_glosses():
    try:
        text = request.args.get('text')
        language = request.args.get('spoken')
        signed = request.args.get('signed')
        if not text or not language or not signed:
            return jsonify({'error': 'Missing required fields: text or spoken or signed'}), 400

        glosses = text_to_gloss(text, language)
        return jsonify({'glosses': glosses})
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port_number = 3004
    app.run(debug=True, host='0.0.0.0', port=port_number)
    print(f"Server listening on port: {port_number}")

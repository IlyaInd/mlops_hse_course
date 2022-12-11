from core.models import app, init_models_in_db

if __name__ == '__main__':
    print('=' * 20, 'START UPLOAD TWO INITIAL MODELS', '=' * 20)
    init_models_in_db()
    print('=' * 20, 'init_models_in_db FINISHED SUCCESSFULLY', '=' * 20)
    app.run(debug=False, host='0.0.0.0', port=5001)

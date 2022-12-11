from core.models import app, init_models_in_db

if __name__ == '__main__':
    init_models_in_db()
    app.run(debug=True, host='0.0.0.0', port=5001)

import time
from core.models import app, init_models_in_db
from core.db import engine, Base

if __name__ == '__main__':
    time.sleep(2)  # чтобы успел запуститься postgres
    print('=' * 20, 'CREATE TABLE MODELS', '=' * 20)
    Base.metadata.create_all(engine)
    print('=' * 20, 'START UPLOAD TWO INITIAL MODELS', '=' * 20)
    engine.connect()
    init_models_in_db()
    print('=' * 20, 'init_models_in_db FINISHED SUCCESSFULLY', '=' * 20)
    app.run(debug=False, host='0.0.0.0', port=5001)

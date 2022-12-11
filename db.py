from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE = {
    'drivername': 'postgresql',
    'host': 'localhost',
    'port': '5432',
    'username': 'root',
    'password': 'root',
    'database': 'postgres'
}

engine = create_engine(URL.create(**DATABASE))

Base = declarative_base()
Session = sessionmaker(bind=engine)

class Models(Base):
    __tablename__ = "models"

    model_id = Column(Integer, primary_key=True)
    model_name = Column(String)
    model_binary = Column(LargeBinary)

    def __init__(self, model_id, model_name, model_binary):
        self.model_id = model_id
        self.model_name = model_name
        self.model_binary = model_binary


if __name__ == '__main__':
    Base.metadata.create_all(engine)



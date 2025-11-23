from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.models import Base
from core.config import Settings

settings = Settings()

# Configurar el engine con el schema si es necesario
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"options": f"-csearch_path={settings.DATABASE_SCHEMA}"} if hasattr(settings, 'DATABASE_SCHEMA') else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

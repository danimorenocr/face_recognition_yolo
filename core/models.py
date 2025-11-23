from sqlalchemy import Boolean, Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Usuario(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    access = Column(Boolean, default=False)

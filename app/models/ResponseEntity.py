from sqlalchemy import Column, Integer, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ResponseEntity(Base):
    __tablename__ = 'qchat'

    id = Column(Integer, primary_key=True, index=True)
    A1 = Column(Integer)
    A2 = Column(Integer)
    A3 = Column(Integer)
    A4 = Column(Integer)
    A5 = Column(Integer)
    A6 = Column(Integer)
    A7 = Column(Integer)
    A8 = Column(Integer)
    A9 = Column(Integer)
    A10 = Column(Integer)
    Age_Mons = Column(Integer)
    Sex = Column(String)
    Ethnicity = Column(String)
    Jaundice = Column(String)
    Family_mem_with_ASD = Column(String)
    Class_ASD_Traits = Column(String, nullable=True, default="")
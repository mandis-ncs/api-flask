from models.ResponseItem import ResponseItem
from models.ResponseEntity import ResponseEntity, Base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine

DATABASE_URL = 'sqlite:///./assets/db/qchat.db'
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Cria as tabelas no banco de dados
Base.metadata.create_all(bind=engine)

# Dependência para obter a sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create(db: Session, response: ResponseItem):
    print(response)
    try:
        print(response.A1)
        obj = ResponseEntity(
            A1=response.A1,
            A2=response.A2,
            A3=response.A3,
            A4=response.A4,
            A5=response.A5,
            A6=response.A6,
            A7=response.A7,
            A8=response.A8,
            A9=response.A9,
            A10=response.A10,
            Age_Mons=response.Age_Mons,
            Sex=response.Sex,
            Ethnicity=response.Ethnicity,
            Jaundice=response.Jaundice,
            Family_mem_with_ASD=response.Family_mem_with_ASD,
            Class_ASD_Traits=response.Class_ASD_Traits
        )
        db.add(obj)
        db.commit()
        db.refresh(obj)
        
        return obj.id
    
    except Exception as e:
        raise RuntimeError(f'\nErro ao salvar os dados no banco: \n{e}\n')
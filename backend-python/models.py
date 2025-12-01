from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "sys_user"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    avatar = Column(String(255), nullable=True)
    role = Column(String(20), default="user") # user, admin
    created_at = Column(DateTime, default=datetime.now)

    records = relationship("IdentificationRecord", back_populates="user")

class WasteCategory(Base):
    __tablename__ = "waste_category"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    category_type = Column(String(50), nullable=False) # Recyclable, Hazardous, Wet, Dry
    explanation = Column(Text, nullable=True)
    disposal_tips = Column(Text, nullable=True) # JSON string or separated by |

class IdentificationRecord(Base):
    __tablename__ = "identification_record"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("sys_user.id"))
    image_url = Column(Text, nullable=True) # Base64 or URL
    waste_name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    explanation = Column(Text, nullable=True)
    create_time = Column(DateTime, default=datetime.now)

    user = relationship("User", back_populates="records")

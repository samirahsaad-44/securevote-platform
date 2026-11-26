import os
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Dict, Optional

# 1. إعدادات اتصال قاعدة البيانات PostgreSQL (يدعم البيئة السحابية والمحلية)
DEFAULT_LOCAL_DB = "postgresql+psycopg2://postgres:Samar%40236@localhost:5432/securevote_db"
raw_db_url = os.getenv("DATABASE_URL", DEFAULT_LOCAL_DB).strip()

if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif raw_db_url.startswith("postgresql://") and "+psycopg2" not in raw_db_url:
    raw_db_url = raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

is_remote_db = "localhost" not in raw_db_url and "127.0.0.1" not in raw_db_url
connect_args = {"sslmode": "require"} if is_remote_db and "sslmode=" not in raw_db_url else {}

engine = create_engine(raw_db_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. إنشاء تطبيق FastAPI
app = FastAPI(title="SecureVote AI & Accessible API", version="3.1-Beta")

# 3. تفعيل إعدادات الـ CORS للسماح بالوصول من الواجهة المنشورة والجوالات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. جداول قاعدة البيانات (Polls/Surveys & Votes)
class PollModel(Base):
    __tablename__ = "polls"

    id = Column(Integer, primary_key=True, index=True)
    poll_type = Column(String, nullable=False, default="poll")  # 'poll' أو 'survey'
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    questions_data = Column(Text, nullable=False)

class VoteModel(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, nullable=False)
    voter_name = Column(String, nullable=False)
    voter_email = Column(String, nullable=False)
    voter_phone = Column(String, nullable=False)
    answers_data = Column(Text, nullable=False)
    device_hash = Column(String, nullable=False)
    ai_human_score = Column(Float, nullable=False, default=100.0)
    used_accessibility = Column(Boolean, nullable=False, default=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 5. نماذج Pydantic
class QuestionItem(BaseModel):
    question: str
    options: List[str]

class PollCreate(BaseModel):
    poll_type: str
    title: str
    description: Optional[str] = None
    questions: List[QuestionItem]

class PollResponse(BaseModel):
    id: int
    poll_type: str
    title: str
    description: Optional[str] = None
    questions: List[QuestionItem]

    class Config:
        from_attributes = True

class BotTelemetry(BaseModel):
    mouse_moves: int
    key_strokes: int
    time_spent_sec: float
    is_webdriver: bool
    accessibility_active: Optional[bool] = False

class VoteCreate(BaseModel):
    poll_id: int
    voter_name: str
    voter_email: str
    voter_phone: str
    answers: Dict[str, str]
    device_hash: str
    telemetry: BotTelemetry

class VoteResponse(BaseModel):
    id: int
    poll_id: int
    voter_name: str
    ai_human_score: float

    class Config:
        from_attributes = True

# 6. محرك الذكاء الاصطناعي لتحليل السلوك وكشف البوتات (مع مراعاة ذوي الاحتياجات الخاصة)
def analyze_bot_behavior(telemetry: BotTelemetry) -> float:
    if telemetry.is_webdriver:
        return 0.0

    score = 100.0

    if telemetry.accessibility_active:
        if telemetry.time_spent_sec < 1.5:
            score -= 60.0
        return max(0.0, score)

    if telemetry.time_spent_sec < 2.0:
        score -= 60.0
    elif telemetry.time_spent_sec < 4.0:
        score -= 25.0

    if telemetry.mouse_moves == 0:
        score -= 35.0
    elif telemetry.mouse_moves < 3:
        score -= 15.0

    if telemetry.key_strokes < 5:
        score -= 30.0

    return max(0.0, score)

# 7. مسارات الـ API
@app.get("/")
def read_root():
    return {"status": "online", "message": "SecureVote AI & Accessible API يعمل بنجاح!"}

@app.get("/polls", response_model=List[PollResponse])
def get_polls(db: Session = Depends(get_db)):
    polls = db.query(PollModel).all()
    result = []
    for p in polls:
        q_list = json.loads(p.questions_data) if p.questions_data else []
        result.append(PollResponse(
            id=p.id,
            poll_type=p.poll_type,
            title=p.title,
            description=p.description,
            questions=q_list
        ))
    return result

@app.post("/polls", response_model=PollResponse)
def create_poll(poll: PollCreate, db: Session = Depends(get_db)):
    if poll.poll_type == "poll" and len(poll.questions) != 1:
        raise HTTPException(status_code=400, detail="التصويت يجب أن يحتوي على سؤال واحد فقط.")
    if poll.poll_type == "survey" and len(poll.questions) < 1:
        raise HTTPException(status_code=400, detail="الاستبيان يجب أن يحتوي على سؤال واحد على الأقل.")

    questions_json = json.dumps([q.dict() for q in poll.questions], ensure_ascii=False)
    db_poll = PollModel(
        poll_type=poll.poll_type,
        title=poll.title,
        description=poll.description,
        questions_data=questions_json
    )
    db.add(db_poll)
    db.commit()
    db.refresh(db_poll)

    return PollResponse(
        id=db_poll.id,
        poll_type=db_poll.poll_type,
        title=db_poll.title,
        description=db_poll.description,
        questions=poll.questions
    )

@app.post("/votes", response_model=VoteResponse)
def cast_vote(vote: VoteCreate, db: Session = Depends(get_db)):
    human_score = analyze_bot_behavior(vote.telemetry)
    if human_score < 50.0:
        raise HTTPException(
            status_code=403,
            detail=f"تم الحظر بواسطة نظام الذكاء الاصطناعي (AI Anti-Bot)! تم رصد سلوك آلي مشبوه (نسبة البشرية: {human_score}%)."
        )

    clean_email = vote.voter_email.strip().lower()
    clean_phone = vote.voter_phone.strip()

    existing_vote = db.query(VoteModel).filter(
        VoteModel.poll_id == vote.poll_id,
        or_(
            VoteModel.device_hash == vote.device_hash,
            VoteModel.voter_email == clean_email,
            VoteModel.voter_phone == clean_phone
        )
    ).first()

    if existing_vote:
        reason = "بصمة الجهاز"
        if existing_vote.voter_email == clean_email:
            reason = "البريد الإلكتروني"
        elif existing_vote.voter_phone == clean_phone:
            reason = "رقم الجوال"
        raise HTTPException(
            status_code=400,
            detail=f"تم رفض العملية! تم استخدام ({reason}) مسبقاً للمشاركة في هذا التصويت/الاستبيان."
        )

    answers_json = json.dumps(vote.answers, ensure_ascii=False)
    db_vote = VoteModel(
        poll_id=vote.poll_id,
        voter_name=vote.voter_name.strip(),
        voter_email=clean_email,
        voter_phone=clean_phone,
        answers_data=answers_json,
        device_hash=vote.device_hash,
        ai_human_score=human_score,
        used_accessibility=bool(vote.telemetry.accessibility_active)
    )
    db.add(db_vote)
    db.commit()
    db.refresh(db_vote)

    return VoteResponse(
        id=db_vote.id,
        poll_id=db_vote.poll_id,
        voter_name=db_vote.voter_name,
        ai_human_score=db_vote.ai_human_score
    )

@app.get("/polls/{poll_id}/results")
def get_poll_results(poll_id: int, db: Session = Depends(get_db)):
    poll = db.query(PollModel).filter(PollModel.id == poll_id).first()
    if not poll:
        raise HTTPException(status_code=404, detail="التصويت/الاستبيان غير موجود")

    questions = json.loads(poll.questions_data) if poll.questions_data else []
    votes = db.query(VoteModel).filter(VoteModel.poll_id == poll_id).all()

    questions_results = []
    for q in questions:
        q_text = q["question"]
        opts = q["options"]
        counts = {opt: 0 for opt in opts}

        for v in votes:
            v_answers = json.loads(v.answers_data)
            chosen = v_answers.get(q_text)
            if chosen in counts:
                counts[chosen] += 1

        questions_results.append({
            "question": q_text,
            "option_counts": counts
        })

    accessibility_users_count = sum(1 for v in votes if v.used_accessibility)

    return {
        "poll_id": poll.id,
        "poll_type": poll.poll_type,
        "title": poll.title,
        "description": poll.description,
        "total_votes": len(votes),
        "accessibility_users_count": accessibility_users_count,
        "questions_results": questions_results,
        "votes_list": [
            {
                "voter_name": v.voter_name,
                "voter_email": v.voter_email,
                "answers": json.loads(v.answers_data),
                "device_hash": v.device_hash,
                "ai_human_score": v.ai_human_score,
                "used_accessibility": v.used_accessibility
            } for v in votes
        ]
    }
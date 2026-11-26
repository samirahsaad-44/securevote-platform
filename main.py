import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. إعداد رابط قاعدة البيانات السحابية (Neon PostgreSQL) مع دعم SSL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./securevote_local.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args: dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif "sslmode=" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 2. نماذج الجداول في قاعدة البيانات
class PollRecord(Base):
    __tablename__ = "polls"

    id = Column(Integer, primary_key=True, index=True)
    poll_type = Column(String(20), nullable=False, default="poll")  # "poll" أو "survey"
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True, default="")
    questions_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class VoteRecord(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, ForeignKey("polls.id"), nullable=False, index=True)
    voter_name = Column(String(150), nullable=False)
    voter_email = Column(String(180), nullable=False, index=True)
    voter_phone = Column(String(50), nullable=False, index=True)
    device_hash = Column(String(128), nullable=False, index=True)
    answers_json = Column(Text, nullable=False)
    used_accessibility = Column(Boolean, default=False)
    ai_human_score = Column(Integer, default=100)
    bot_risk_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


# 3. تهيئة تطبيق FastAPI و CORS للنشر السحابي
app = FastAPI(
    title="SecureVote AI & Accessible API",
    version="2.0.0",
    description="Backend API for fraud-resistant, accessible voting and surveys.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. مخططات البيانات (Pydantic Schemas) المطابقة للواجهة
class QuestionItem(BaseModel):
    question: str = Field(..., min_length=1)
    options: list[str] = Field(..., min_length=1)


class PollCreatePayload(BaseModel):
    poll_type: str = Field(default="poll")
    title: str = Field(..., min_length=1)
    description: Optional[str] = ""
    questions: list[QuestionItem] = Field(..., min_length=1)


class TelemetryPayload(BaseModel):
    mouse_moves: int = 0
    key_strokes: int = 0
    time_spent_sec: float = 0.0
    is_webdriver: bool = False
    accessibility_active: bool = False


class VoteSubmitPayload(BaseModel):
    poll_id: int
    voter_name: str = Field(..., min_length=1)
    voter_email: str = Field(..., min_length=3)
    voter_phone: str = Field(..., min_length=4)
    answers: dict[str, str] = Field(default_factory=dict)
    device_hash: str = Field(default="")
    telemetry: TelemetryPayload = Field(default_factory=TelemetryPayload)


def evaluate_human_score(telemetry: TelemetryPayload) -> tuple[int, float, str]:
    """حساب نسبة البشرية (0 - 100) ومؤشر مخاطر البوتات."""
    if telemetry.is_webdriver:
        return 5, 0.95, "تم اكتشاف بيئة تشغيل آلية (WebDriver)"

    risk = 0.04
    if telemetry.time_spent_sec < 1.5:
        risk += 0.45
    if telemetry.mouse_moves == 0 and telemetry.key_strokes == 0 and not telemetry.accessibility_active:
        risk += 0.35
    if telemetry.accessibility_active:
        risk = max(0.02, risk - 0.02)

    clamped_risk = round(min(max(risk, 0.01), 0.99), 2)
    human_score = int(round((1.0 - clamped_risk) * 100))
    return human_score, clamped_risk, "تحقق سلوكي ناجح"


def serialize_poll(record: PollRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "poll_type": record.poll_type,
        "title": record.title,
        "description": record.description or "",
        "questions": json.loads(record.questions_json),
    }


def ensure_sample_poll(db) -> None:
    if db.query(PollRecord).count() == 0:
        sample = PollRecord(
            poll_type="poll",
            title="ما مدى رضاك عن سهولة الوصول وإتاحة الخدمات الرقمية؟",
            description="تصويت تجريبي جاهز لاختبار المنصة وحفظ النتائج في قاعدة البيانات.",
            questions_json=json.dumps(
                [
                    {
                        "question": "ما مدى رضاك عن سهولة الوصول وإتاحة الخدمات الرقمية؟",
                        "options": ["ممتاز وسهل جداً", "جيد ويحتاج تحسينات بسيطة", "يحتاج تطوير إضافي"],
                    }
                ],
                ensure_ascii=False,
            ),
        )
        db.add(sample)
        db.commit()


# 5. نقاط النهاية (Endpoints)
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "SecureVote AI & Accessible API يعمل بنجاح",
    }


@app.get("/health")
def health_check():
    db = SessionLocal()
    try:
        ensure_sample_poll(db)
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    finally:
        db.close()


@app.get("/polls")
def list_polls():
    db = SessionLocal()
    try:
        ensure_sample_poll(db)
        records = db.query(PollRecord).order_by(PollRecord.id.asc()).all()
        return [serialize_poll(r) for r in records]
    finally:
        db.close()


@app.post("/polls")
def create_poll_or_survey(payload: PollCreatePayload):
    cleaned_questions = []
    for q in payload.questions:
        q_text = q.question.strip()
        opts = [opt.strip() for opt in q.options if opt.strip()]
        if q_text and len(opts) >= 1:
            cleaned_questions.append({"question": q_text, "options": opts})

    if not cleaned_questions:
        raise HTTPException(status_code=400, detail="يجب إدخال سؤال واحد وخيار واحد على الأقل.")

    normalized_type = "survey" if payload.poll_type.strip().lower() == "survey" else "poll"

    db = SessionLocal()
    try:
        record = PollRecord(
            poll_type=normalized_type,
            title=payload.title.strip(),
            description=(payload.description or "").strip(),
            questions_json=json.dumps(cleaned_questions, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return serialize_poll(record)
    finally:
        db.close()


@app.post("/votes")
def submit_vote(payload: VoteSubmitPayload, request: Request):
    _ = request.client.host if request.client else "unknown"
    email_norm = payload.voter_email.strip().lower()
    phone_norm = payload.voter_phone.strip()
    device_norm = (payload.device_hash or "no-device-hash").strip()

    human_score, risk_score, reason = evaluate_human_score(payload.telemetry)
    if risk_score >= 0.80:
        raise HTTPException(
            status_code=403,
            detail=f"تم رفض المشاركة بواسطة نظام الحماية من البوتات ({reason}).",
        )

    db = SessionLocal()
    try:
        poll = db.query(PollRecord).filter(PollRecord.id == payload.poll_id).first()
        if not poll:
            raise HTTPException(status_code=404, detail="التصويت أو الاستبيان غير موجود.")

        questions = json.loads(poll.questions_json)
        for q in questions:
            q_title = q["question"]
            if q_title not in payload.answers or not str(payload.answers[q_title]).strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"يرجى اختيار إجابة للسؤال: {q_title}",
                )

        duplicate_email = (
            db.query(VoteRecord)
            .filter(VoteRecord.poll_id == poll.id, VoteRecord.voter_email == email_norm)
            .first()
        )
        if duplicate_email:
            raise HTTPException(
                status_code=409,
                detail="تم تسجيل مشاركة سابقة بهذا البريد الإلكتروني في هذا الاستبيان/التصويت.",
            )

        duplicate_phone = (
            db.query(VoteRecord)
            .filter(VoteRecord.poll_id == poll.id, VoteRecord.voter_phone == phone_norm)
            .first()
        )
        if duplicate_phone:
            raise HTTPException(
                status_code=409,
                detail="تم تسجيل مشاركة سابقة برقم الجوال هذا في هذا الاستبيان/التصويت.",
            )

        if device_norm != "no-device-hash":
            duplicate_device = (
                db.query(VoteRecord)
                .filter(VoteRecord.poll_id == poll.id, VoteRecord.device_hash == device_norm)
                .first()
            )
            if duplicate_device:
                raise HTTPException(
                    status_code=409,
                    detail="تم تسجيل مشاركة سابقة من هذا الجهاز في هذا الاستبيان/التصويت.",
                )

        vote = VoteRecord(
            poll_id=poll.id,
            voter_name=payload.voter_name.strip(),
            voter_email=email_norm,
            voter_phone=phone_norm,
            device_hash=device_norm,
            answers_json=json.dumps(payload.answers, ensure_ascii=False),
            used_accessibility=bool(payload.telemetry.accessibility_active),
            ai_human_score=human_score,
            bot_risk_score=risk_score,
        )
        db.add(vote)
        db.commit()
        db.refresh(vote)

        return {
            "status": "recorded",
            "vote_id": vote.id,
            "ai_human_score": vote.ai_human_score,
            "bot_risk_score": vote.bot_risk_score,
        }
    finally:
        db.close()


@app.get("/polls/{poll_id}/results")
def get_poll_results(poll_id: int):
    db = SessionLocal()
    try:
        poll = db.query(PollRecord).filter(PollRecord.id == poll_id).first()
        if not poll:
            raise HTTPException(status_code=404, detail="التصويت أو الاستبيان غير موجود.")

        questions = json.loads(poll.questions_json)
        votes = (
            db.query(VoteRecord)
            .filter(VoteRecord.poll_id == poll.id)
            .order_by(VoteRecord.id.desc())
            .all()
        )

        questions_results = []
        for q in questions:
            q_text = q["question"]
            counts = {opt: 0 for opt in q["options"]}
            for v in votes:
                v_answers = json.loads(v.answers_json)
                chosen = v_answers.get(q_text)
                if chosen in counts:
                    counts[chosen] += 1
            questions_results.append({
                "question": q_text,
                "option_counts": counts,
            })

        accessibility_count = sum(1 for v in votes if v.used_accessibility)
        votes_list = [
            {
                "voter_name": v.voter_name,
                "voter_email": v.voter_email,
                "used_accessibility": v.used_accessibility,
                "ai_human_score": v.ai_human_score,
            }
            for v in votes
        ]

        return {
            "poll_id": poll.id,
            "title": poll.title,
            "poll_type": poll.poll_type,
            "total_votes": len(votes),
            "accessibility_users_count": accessibility_count,
            "questions_results": questions_results,
            "votes_list": votes_list,
        }
    finally:
        db.close()

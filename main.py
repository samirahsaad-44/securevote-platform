import os
from datetime import datetime, timezone
from typing import Optional

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

connect_args = {}
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
class Poll(Base):
    __tablename__ = "polls"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    options_json = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Vote(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, ForeignKey("polls.id"), nullable=False, index=True)
    voter_role = Column(String(50), nullable=False)
    voter_identifier = Column(String(120), nullable=False)
    selected_option = Column(String(255), nullable=False)
    bot_risk_score = Column(Float, default=0.0)
    telemetry_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


# 3. تهيئة تطبيق FastAPI و CORS للنشر السحابي
app = FastAPI(
    title="SecureVote AI & Accessible API",
    version="1.0.0",
    description="Backend API for fraud-resistant, accessible voting and surveys.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكن تقييده برابط Vercel/Netlify بعد النشر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. مخططات البيانات (Pydantic Schemas)
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    role: str = Field(default="voter")


class PollCreateRequest(BaseModel):
    title: str = Field(..., min_length=2)
    description: Optional[str] = ""
    options: list[str] = Field(..., min_length=2)


class TelemetryData(BaseModel):
    pointer_events: int = 0
    key_events: int = 0
    dwell_time_ms: int = 0
    accessibility_used: bool = False


class VoteSubmitRequest(BaseModel):
    poll_id: int = Field(default=1)
    voter_role: str = Field(default="voter")
    voter_identifier: str = Field(..., min_length=1)
    selected_option: str = Field(..., min_length=1)
    telemetry: TelemetryData = Field(default_factory=TelemetryData)


def compute_bot_risk(telemetry: TelemetryData) -> tuple[float, str]:
    """تقييم مبدئي لمخاطر السلوك الآلي بناءً على بيانات التفاعل."""
    score = 0.05
    reasons = []

    if telemetry.dwell_time_ms < 400:
        score += 0.45
        reasons.append("زمن استجابة سريع جداً")
    elif telemetry.dwell_time_ms >= 1200:
        score -= 0.02
        reasons.append("زمن تفاعل بشري طبيعي")

    if telemetry.pointer_events == 0 and telemetry.key_events == 0:
        score += 0.35
        reasons.append("غياب أحداث المؤشر ولوحة المفاتيح")

    if telemetry.accessibility_used:
        score = max(0.02, score - 0.10)
        reasons.append("تفعيل أدوات الإتاحة")

    clamped_score = round(min(max(score, 0.01), 0.99), 2)
    summary = " | ".join(reasons) if reasons else "تفاعل طبيعي"
    return clamped_score, summary


def ensure_default_poll(db) -> Poll:
    poll = db.query(Poll).filter(Poll.is_active == True).first()  # noqa: E712
    if not poll:
        import json

        poll = Poll(
            title="استطلاع تجربة المستخدم والخدمات الرقمية",
            description="تصويت تجريبي حي لاختبار المنصة والاتصال بقاعدة البيانات السحابية.",
            options_json=json.dumps(
                ["واجهة سهلة ومتاحة للجميع", "سرعة استجابة عالية", "أمان وموثوقية البيانات"],
                ensure_ascii=False,
            ),
            is_active=True,
        )
        db.add(poll)
        db.commit()
        db.refresh(poll)
    return poll


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
        ensure_default_poll(db)
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    finally:
        db.close()


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    normalized_role = payload.role.strip().lower()
    if normalized_role not in {"admin", "voter", "auditor"}:
        normalized_role = "voter"
    return {
        "authenticated": True,
        "username": payload.username.strip(),
        "role": normalized_role,
        "token_type": "session",
    }


@app.get("/api/polls/active")
def get_active_poll():
    import json

    db = SessionLocal()
    try:
        poll = ensure_default_poll(db)
        votes = db.query(Vote).filter(Vote.poll_id == poll.id).all()
        options = json.loads(poll.options_json)
        counts = {opt: 0 for opt in options}
        for v in votes:
            if v.selected_option in counts:
                counts[v.selected_option] += 1
        return {
            "id": poll.id,
            "title": poll.title,
            "description": poll.description,
            "options": options,
            "total_votes": len(votes),
            "counts": counts,
        }
    finally:
        db.close()


@app.post("/api/polls")
def create_poll(payload: PollCreateRequest):
    import json

    cleaned_options = [opt.strip() for opt in payload.options if opt.strip()]
    if len(cleaned_options) < 2:
        raise HTTPException(status_code=400, detail="يجب إدخال خيارين على الأقل.")

    db = SessionLocal()
    try:
        db.query(Poll).update({Poll.is_active: False})
        new_poll = Poll(
            title=payload.title.strip(),
            description=(payload.description or "").strip(),
            options_json=json.dumps(cleaned_options, ensure_ascii=False),
            is_active=True,
        )
        db.add(new_poll)
        db.commit()
        db.refresh(new_poll)
        return {"id": new_poll.id, "title": new_poll.title, "options": cleaned_options}
    finally:
        db.close()


@app.post("/api/votes")
def submit_vote(payload: VoteSubmitRequest, request: Request):
    db = SessionLocal()
    try:
        poll = ensure_default_poll(db)
        target_poll = db.query(Poll).filter(Poll.id == payload.poll_id).first() or poll

        existing_vote = (
            db.query(Vote)
            .filter(
                Vote.poll_id == target_poll.id,
                Vote.voter_identifier == payload.voter_identifier.strip(),
            )
            .first()
        )
        if existing_vote:
            raise HTTPException(
                status_code=409,
                detail="تم تسجيل صوت مسبق لهذا المعرّف في هذا الاستطلاع.",
            )

        risk_score, summary = compute_bot_risk(payload.telemetry)
        if risk_score >= 0.80:
            raise HTTPException(
                status_code=429,
                detail=f"تم إيقاف الطلب مؤقتاً بواسطة نظام الحماية من البوتات ({summary}).",
            )

        _ = request.client.host if request.client else "unknown"
        vote = Vote(
            poll_id=target_poll.id,
            voter_role=payload.voter_role.strip().lower(),
            voter_identifier=payload.voter_identifier.strip(),
            selected_option=payload.selected_option.strip(),
            bot_risk_score=risk_score,
            telemetry_summary=summary,
        )
        db.add(vote)
        db.commit()
        db.refresh(vote)

        return {
            "status": "recorded",
            "vote_id": vote.id,
            "bot_risk_score": vote.bot_risk_score,
            "telemetry_summary": vote.telemetry_summary,
        }
    finally:
        db.close()


@app.get("/api/votes/summary")
def votes_summary():
    db = SessionLocal()
    try:
        poll = ensure_default_poll(db)
        votes = db.query(Vote).filter(Vote.poll_id == poll.id).order_by(Vote.id.desc()).all()
        avg_risk = (
            round(sum(v.bot_risk_score for v in votes) / len(votes), 2) if votes else 0.0
        )
        recent = [
            {
                "id": v.id,
                "voter_role": v.voter_role,
                "selected_option": v.selected_option,
                "bot_risk_score": v.bot_risk_score,
                "telemetry_summary": v.telemetry_summary,
            }
            for v in votes[:10]
        ]
        return {
            "total_votes": len(votes),
            "average_bot_risk": avg_risk,
            "recent_votes": recent,
        }
    finally:
        db.close()

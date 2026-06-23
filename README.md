# CAPD AI

> **복막투석 AI 기록 검토 시스템 — AI / RAG 파이프라인**

KDIGO 가이드라인 기반 규칙 탐지와 RAG(Retrieval-Augmented Generation)를 결합해 복막투석 환자의 일일 기록에서 이상 징후를 감지하고, 맞춤 후속 질문을 자동 생성하는 AI 모듈입니다.

🔗 [백엔드 레포](https://github.com/KimDimension/backend) · [프론트엔드 레포](https://github.com/KimDimension/frontend)

---

## 파이프라인 개요

```
환자 일일 기록 (입력)
        ↓
1. 규칙 기반 탐지 (KDIGO 가이드라인)
   ├─ 이상 징후 탐지 시 → 직접 플래그
   └─ 모호하거나 정보 부족 시 → RAG 단계로 전달
        ↓
2. RAG — pgvector 유사도 검색
   └─ KDIGO 문서에서 관련 컨텍스트 검색
        ↓
3. LLM (Qwen2.5-3B, LM Studio)
   └─ 컨텍스트 기반 맞춤 후속 질문 생성
        ↓
결과 반환 (백엔드 → 프론트엔드)
```

---

## 주요 기능

- **KDIGO 규칙 엔진** — 복막투석 가이드라인 기반 1차 이상 징후 탐지
- **RAG 파이프라인** — sentence-transformers 임베딩 + pgvector 유사도 검색
- **LLM 추론** — LM Studio에서 Qwen2.5-3B 로컬 서빙, 맞춤 질문 생성
- **GCP 전환 설계** — Cloud Run + Gemini API로 전환 가능한 구조 설계 완료

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.11+ |
| **임베딩** | sentence-transformers |
| **벡터 DB** | pgvector (PostgreSQL 확장) |
| **LLM 서빙** | LM Studio · Qwen2.5-3B |
| **배포** | Docker · (GCP Cloud Run 전환 설계 완료) |

---

## 시작하기

### 요구 사항

- Python 3.11+
- PostgreSQL (pgvector 확장 활성화)
- LM Studio 실행 중 (Qwen2.5-3B 모델 로드)

### 설치

```bash
pip install -r requirements.txt
```

### 환경 변수

```env
DATABASE_URL=postgresql://user:password@localhost:5432/capd
LM_STUDIO_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 실행

```bash
python main.py
```

### Docker

```bash
docker build -t capd-ai .
docker run --env-file .env capd-ai
```

---

## 프로젝트 구조

```
capd-ai/
├── rules/          # KDIGO 가이드라인 기반 규칙 정의
├── rag/            # 문서 임베딩 및 pgvector 검색
├── llm/            # LM Studio 연동 및 프롬프트 관리
├── pipeline.py     # 전체 파이프라인 오케스트레이션
├── main.py         # 진입점
└── requirements.txt
```

---

## 관련 레포

| 레포 | 설명 |
|------|------|
| [backend](https://github.com/KimDimension/backend) | FastAPI 서버 · DB 스키마 · API 엔드포인트 |
| [frontend](https://github.com/KimDimension/frontend) | TypeScript 웹 클라이언트 |

---

## 팀

**KimDimension (CAPD)** — 복막투석 AI 기록 검토 시스템 개발팀

# RAG API
rag-api-doc/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/routes/
│   │   ├── ingest.py        # now returns job_id
│   │   └── query.py
│   ├── core/
│   │   ├── loader.py        # unchanged
│   │   ├── splitter.py      # unchanged
│   │   ├── embeddings.py    # unchanged
│   │   ├── vectorstore.py   # ← PGVector
│   │   ├── llm.py           # unchanged
│   │   └── prompt.py        # unchanged
│   ├── db/
│   │   ├── postgres.py      # ← SQLAlchemy models + init_db
│   │   └── metadata.py      # ← updated helpers
│   ├── services/
│   │   ├── ingest_service.py
│   │   └── rag_service.py   # unchanged
│   ├── models/schemas.py
│   └── worker/
│       ├── celery_app.py    # ← new
│       └── tasks.py         # ← new
├── docker-compose.yml
├── .env
└── requirements.txt
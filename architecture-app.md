
```
Certif-Exam-Generator/
├── architecture-app.md                 ← you are here
├── README.md
├── pyproject.toml
├── .env.example
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                      # lint + pytest on PR
│   │   └── weekly-ingest.yml           # §9
│   └── dependabot.yml
├── src/certgen/
│   ├── __init__.py
│   ├── core/
│   │   ├── settings.py
│   │   ├── model_router.py
│   │   └── cloud_provider.py
│   ├── ingestion/
│   │   ├── sitemap_loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── hasher.py
│   │   ├── delta.py
│   │   ├── embedder.py
│   │   ├── upserter.py
│   │   └── runner.py
│   ├── rag/
│   │   └── hybrid_retriever.py
│   ├── agent/
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── retriever.py
│   │       ├── question_gen.py
│   │       ├── distractor_gen.py
│   │       └── validator.py
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── exams.py
│   │       └── ops.py
│   └── observability/
│       ├── langsmith.py
│       └── logging.py
├── tests/
│   ├── unit/
│   ├── integration/                    # hits a testcontainers Mongo + mocked Pinecone
│   └── fixtures/
├── configs/
│   └── certs/gcp-pca.yaml
├── docker-compose.yml                  # local Mongo for offline dev
└── Dockerfile                          # API image
```

---

### Graph shape

```mermaid
stateDiagram-v2
    [*] --> Planner
    Planner --> Retriever : for each plan
    Retriever --> QuestionGen
    QuestionGen --> DistractorGen
    DistractorGen --> Validator
    Validator --> Accumulate : ok == true
    Validator --> QuestionGen : ok == false && retry < 2
    Validator --> Accumulate : ok == false && retry == 2 (drop & log)
    Accumulate --> Retriever : plans remaining
    Accumulate --> [*] : all plans consumed
```


> The `<lastmod>` tag in enterprise sitemaps is unreliable (entire sitemaps often re-timestamp on cosmetic CMS changes). Content hashing is the source of truth.

```mermaid
sequenceDiagram
    participant GH as GH Actions (weekly)
    participant Run as ingestion.runner
    participant Web as cloud.google.com
    participant Mongo as MongoDB Atlas
    participant Emb as ModelRouter (embeddings)
    participant Pine as Pinecone

    GH->>Run: cron trigger
    Run->>Web: SitemapLoader.fetch()
    Web-->>Run: HTML pages
    Run->>Run: parse + chunk + hash
    loop per chunk
        Run->>Mongo: find_one({_id: chunk_id})
        alt hash matches stored
            Run->>Mongo: $set last_seen_at = now
            Note right of Run: no embedding cost
        else new or hash differs
            Run->>Emb: embed(chunk.content)
            Emb-->>Run: vector
            Run->>Pine: upsert(chunk_id, vector, metadata)
            Run->>Mongo: upsert chunk + parent doc
        end
    end
    Run->>Mongo: mark tombstoned where last_seen_at < run_started_at
    Run->>Pine: delete tombstoned ids
    Run->>Mongo: insert ingestion_runs record
```
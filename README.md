# MiniCascade-RAG


A production-ready Reasoning RAG agent applications with LangGraph integration. Providing a robust foundation for building scalable, secure, and maintainable AI agent services.


## 🌟 Features

- **Production-Ready Architecture**

  - FastAPI for high-performance async API endpoints
  - LangGraph integration for AI agent workflows
  - Langfuse for LLM observability and monitoring
  - Structured logging with environment-specific formatting
  - PostgreSQL for data persistence
  - Docker and Docker Compose support


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL ([see Database setup](#database-setup))
- Docker and Docker Compose (optional)


### Environment Setup
1. Clone the repository:

```bash
git clone https://github.com/zqh2029/MiniCascade-RAG.git
cd MiniCascade-RAG
```

2. Create and activate a virtual environment:



#### uv environment
Enter project directory, use：
```bash
uv sync   # if not have uv use "pip install uv" for install
``` 
to synchronize project environment automatically

### Database setup

1. Create a PostgreSQL database (e.g Supabase or local PostgreSQL)
2. Update the database connection string in your `.env` file:

```bash
POSTGRES_URL="postgresql://:your-db-password@POSTGRES_HOST:POSTGRES_PORT/POSTGRES_DB"
```

3. Launch Qdrant：
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
Rabbitmq
#### latest RabbitMQ 4.x
```bash
docker run -it --rm --name rabbitmq \
    -p 5672:5672 -p 15672:15672 \
    rabbitmq:4-management
```
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4-management
### Deployment use Docker Compose
```bash
docker compose up
```



## API KEY Configure

All the sensitive credentials are placed in a `.env` file that will always sit at the root of your directory, at the same level with the `.env.example` file.

Go to the root of the repository and copy our `.env.example` file as follows:
```shell
cp .env.example .env
```
Now fill it with your credentials, following the suggestions from the next section.

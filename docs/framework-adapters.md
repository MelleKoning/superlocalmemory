# Framework Adapters

Nine Python packages back their framework's native memory interface with the local SLM data root. Install each alongside the target framework; all data stays in the configured data root unless an optional SLM provider, connector, or backup feature is explicitly enabled.

**Prerequisites:** SuperLocalMemory installed in the same environment, Python 3.12 to 3.14, and a supported platform (Apple Silicon macOS, 64-bit Windows, 64-bit Linux).

## Installing an adapter

Three adapters are published on PyPI: LangChain, LlamaIndex and CrewAI. Install them with the `pip install` command shown for each. The other six are not on PyPI yet, so a `pip install <name>` for them fails; install them from the repository folder instead:

```bash
git clone https://github.com/qualixar/superlocalmemory
pip install ./superlocalmemory/ide/integrations/langgraph
```

The folder for each adapter is in the table at the end of this page. The `pip install` line under each of those six headings below is the name the package will have once it is published, and the folder install gives the same package.

---

## LangGraph

**Install:** `pip install ./superlocalmemory/ide/integrations/langgraph` (from a clone of the repository)  
**Requires:** `langgraph >= 1.0.0`  
**Implements:** `BaseStore`  
**Class:** `SuperLocalMemoryStore`

```python
from langgraph_superlocalmemory import SuperLocalMemoryStore

store = SuperLocalMemoryStore()
store.put(("users", "1"), "profile", {"name": "Ada", "role": "engineer"})
item = store.get(("users", "1"), "profile")
results = list(store.search(("users",), filter={"role": "engineer"}))
store.delete(("users", "1"), "profile")
```

Drop in as the `store=` argument to `create_react_agent(...)` or `StateGraph(...).compile(store=store)`.

---

## Semantic Kernel

**Install:** `pip install ./superlocalmemory/ide/integrations/semantic-kernel` (from a clone of the repository)  
**Requires:** `semantic-kernel >= 1.34.0`  
**Implements:** `VectorStore`; `get_collection` returns a `VectorStoreCollection`  
**Class:** `SuperLocalMemoryVectorStore`

```python
from semantic_kernel_superlocalmemory import SuperLocalMemoryVectorStore

store = SuperLocalMemoryVectorStore()
collection = store.get_collection(Doc, collection_name="docs")
await collection.ensure_collection_exists()
await collection.upsert(Doc(id="d1", text="hello world"))
doc = await collection.get("d1")
```

Written against Semantic Kernel 1.44. Uses the post-1.34 `semantic_kernel.data.vector` API; the deprecated `MemoryStoreBase` is not used.

---

## Microsoft Agent Framework

**Install:** `pip install ./superlocalmemory/ide/integrations/agent-framework` (from a clone of the repository)  
**Requires:** `agent-framework-core >= 1.5.0`  
**Implements:** `ContextProvider` and `HistoryProvider`  
**Classes:** `SuperLocalMemoryContextProvider`, `SuperLocalMemoryHistoryProvider`

```python
from agent_framework_superlocalmemory import (
    SuperLocalMemoryContextProvider,
    SuperLocalMemoryHistoryProvider,
)

history = SuperLocalMemoryHistoryProvider()
memory = SuperLocalMemoryContextProvider(max_recall=10)
# agent = ChatAgent(..., context_providers=[memory], history_provider=history)
```

`HistoryProvider` implements `get_messages` / `save_messages` per session.  
`ContextProvider` overrides `before_run` (inject context) and `after_run` (persist turn).

---

## LangChain

**Install:** `pip install langchain-superlocalmemory`  
**Requires:** `langchain-core >= 1.0.0`  
**Implements:** `BaseChatMessageHistory`  
**Class:** `SuperLocalMemoryChatMessageHistory`

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

history = SuperLocalMemoryChatMessageHistory(session_id="my-chat-session")
history.add_messages([
    HumanMessage(content="What is SuperLocalMemory?"),
    AIMessage(content="A local-first memory system for AI assistants."),
])
messages = history.messages   # chronological list
history.clear()
```

---

## LlamaIndex

**Install:** `pip install llama-index-storage-chat-store-superlocalmemory`  
**Requires:** SuperLocalMemory installed in the same virtual environment  
**Implements:** `BaseChatStore`  
**Class:** `SuperLocalMemoryChatStore`

```python
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = SuperLocalMemoryChatStore()
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store, chat_store_key="user-123", token_limit=3000)
# Or directly:
chat_store.add_message("session-1", ChatMessage(role=MessageRole.USER, content="Hi"))
```

---

## CrewAI

**Install:** `pip install crewai-superlocalmemory`  
**Requires:** `crewai >= 1.14.6`  
**Implements:** `StorageBackend`  
**Class:** `SuperLocalMemoryBackend`

```python
from crewai_superlocalmemory import SuperLocalMemoryBackend

backend = SuperLocalMemoryBackend()
# Pass as storage_backend= in your CrewAI memory configuration.
```

Each `MemoryRecord` is stored as one SLM memory with `session_id = "crewai-rec:<id>"`. The backend supports scoped search and cosine-ranked retrieval over stored embeddings.

---

## AutoGen

**Install:** `pip install ./superlocalmemory/ide/integrations/autogen` (from a clone of the repository)  
**Requires:** `autogen-agentchat >= 0.7.5`  
**Implements:** `Memory` (autogen-core)  
**Class:** `SuperLocalMemoryMemory`

```python
from autogen_superlocalmemory import SuperLocalMemoryMemory
from autogen_core.memory import MemoryContent, MemoryMimeType

mem = SuperLocalMemoryMemory()
await mem.add(MemoryContent(content="Ada prefers dark mode.", mime_type=MemoryMimeType.TEXT))
result = await mem.query("Ada preferences")
# agent = AssistantAgent("assistant", memory=[mem], model_client=...)
```

> For projects using **Microsoft Agent Framework** (`agent-framework-core`), prefer the `agent-framework-superlocalmemory` adapter instead.

---

## Google ADK

**Install:** `pip install ./superlocalmemory/ide/integrations/google-adk` (from a clone of the repository)  
**Requires:** `google-adk >= 2.5.0`  
**Implements:** `BaseMemoryService`  
**Class:** `SuperLocalMemoryService`

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google_adk_superlocalmemory import SuperLocalMemoryService

service = SuperLocalMemoryService()
runner = Runner(agent=agent, app_name="my-app", memory_service=service)
# After a session runs, events are stored and retrievable via SLM CLI, MCP, and dashboard.
```

Implements `add_session_to_memory` (idempotent upsert per session) and `search_memory` (semantic recall). Custom database path: `SuperLocalMemoryService(db_path="/path/to/memory.db")`.

---

## OpenAI Agents

**Install:** `pip install ./superlocalmemory/ide/integrations/openai-agents` (from a clone of the repository)  
**Requires:** `openai-agents >= 0.18.3`  
**Implements:** `SessionABC`  
**Class:** `SLMSession`

```python
from agents import Runner
from openai_agents_superlocalmemory import SLMSession

session = SLMSession(session_id="user-42-conv-7")
runner = Runner(session=session)
result = await runner.run("What is the capital of France?")
```

Manages the ordered `TResponseInputItem` conversation history for one `session_id`. Custom path: `SLMSession(session_id="s1", db_path="/path/to/memory.db")`.

---

## All Adapters at a Glance

| Framework | Package | On PyPI | Folder in `ide/integrations/` | Interface | Class |
|---|---|:---:|---|---|---|
| LangGraph | `langgraph-superlocalmemory` | no | `langgraph` | `BaseStore` | `SuperLocalMemoryStore` |
| Semantic Kernel | `semantic-kernel-superlocalmemory` | no | `semantic-kernel` | `VectorStore` | `SuperLocalMemoryVectorStore` |
| Microsoft Agent Framework | `agent-framework-superlocalmemory` | no | `agent-framework` | `ContextProvider` / `HistoryProvider` | `SuperLocalMemoryContextProvider`, `SuperLocalMemoryHistoryProvider` |
| LangChain | `langchain-superlocalmemory` | yes | `langchain` | `BaseChatMessageHistory` | `SuperLocalMemoryChatMessageHistory` |
| LlamaIndex | `llama-index-storage-chat-store-superlocalmemory` | yes | `llamaindex` | `BaseChatStore` | `SuperLocalMemoryChatStore` |
| CrewAI | `crewai-superlocalmemory` | yes | `crewai` | `StorageBackend` | `SuperLocalMemoryBackend` |
| AutoGen | `autogen-superlocalmemory` | no | `autogen` | `Memory` | `SuperLocalMemoryMemory` |
| Google ADK | `google-adk-superlocalmemory` | no | `google-adk` | `BaseMemoryService` | `SuperLocalMemoryService` |
| OpenAI Agents | `openai-agents-superlocalmemory` | no | `openai-agents` | `SessionABC` | `SLMSession` |

Each adapter folder has its own README with extended usage examples, prerequisites, and scope notes.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

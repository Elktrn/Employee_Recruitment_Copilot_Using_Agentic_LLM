#!/usr/bin/env python3
"""
Agentic Conversational CV — Dual‑Mode Chat Agent (Employee / Employer)

A single‑file, production‑ready prototype demonstrating:
- Agentic interaction with thoughtful, progressive questioning
- Role‑separated journeys (Employee vs Employer)
- Structured tool/function calling for profile CRUD + search
- Lightweight memory
- Few‑shot prompting + role prompting
- Simple local JSON persistence
- Minimal CLI interface (no external web framework required)

Run:
  export OPENAI_API_KEY=...  # or set via .env
  pip install -q langchain langchain-openai langgraph pydantic python-dotenv rich
  python app.py --mode employee --name "Ada Lovelace"
  python app.py --mode employer

Notes:
- This is intentionally compact yet expressive. Replace the model name as you prefer.
- The agent uses tool-calling to save/search profiles and a LangGraph loop for turn-taking.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TypedDict

from dotenv import load_dotenv
load_dotenv()

# === LangChain / OpenAI / LangGraph ===
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# -----------------------------
# Data Layer (Tiny JSON Store)
# -----------------------------
DB_PATH = os.environ.get("PROFILE_DB", "profiles.json")

@dataclass
class DB:
    path: str = DB_PATH

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save(self, data: Dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def upsert_profile(self, profile: Dict[str, Any]):
        data = self._load()
        name = (profile.get("name") or "").strip()
        if not name:
            raise ValueError("Profile must include a non-empty 'name'.")
        data[name] = {**data.get(name, {}), **profile}
        self._save(data)
        return data[name]

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        return self._load().get(name)

    def search(self, text: Optional[str] = None, title: Optional[str] = None, skill: Optional[str] = None) -> List[Dict[str, Any]]:
        data = self._load()
        results = []
        for p in data.values():
            if title and title.lower() not in (p.get("title") or "").lower():
                continue
            if skill:
                skills = " ".join(p.get("skills", [])) if isinstance(p.get("skills"), list) else str(p.get("skills", ""))
                if skill.lower() not in skills.lower():
                    continue
            if text:
                hay = json.dumps(p, ensure_ascii=False).lower()
                if text.lower() not in hay:
                    continue
            results.append(p)
        return results

DBI = DB()

# -----------------------------
# Profile Schema (for tools)
# -----------------------------
class Experience(BaseModel):
    role: str = Field(..., description="Job title or role, e.g., 'Senior Data Scientist'")
    company: str = Field(..., description="Organization name")
    start: str = Field(..., description="Start date, e.g., '2021-05'")
    end: str = Field(..., description="End date or 'present'")
    achievements: List[str] = Field(default_factory=list, description="Notable outcomes, bullet style")

class Profile(BaseModel):
    name: str = Field(..., description="Full name of the professional")
    title: Optional[str] = Field(None, description="Professional headline, e.g., 'ML Engineer' or 'Product Manager'")
    location: Optional[str] = Field(None, description="City, Country")
    summary: Optional[str] = Field(None, description="Short professional bio/summary")
    skills: List[str] = Field(default_factory=list, description="Key skills, e.g., ['Python', 'Leadership']")
    experiences: List[Experience] = Field(default_factory=list, description="Work experience entries")
    education: List[str] = Field(default_factory=list, description="Education highlights")
    links: List[str] = Field(default_factory=list, description="Portfolio/GitHub/LinkedIn URLs")

# -----------------------------
# Tools (function calling)
# -----------------------------

def _tool_upsert_profile(profile: Profile) -> Dict[str, Any]:
    """Create or update a profile in the JSON store. Returns the stored profile."""
    return DBI.upsert_profile(profile.model_dump())

UpsertProfile = StructuredTool.from_function(
    name="save_profile",
    description=(
        "Create or update a professional profile in persistent storage. "
        "Use this every time you collect enough new information (even partial updates)."
    ),
    func=_tool_upsert_profile,
    args_schema=Profile,
)

class SearchArgs(BaseModel):
    text: Optional[str] = Field(None, description="Full-text contains filter across the entire profile JSON")
    title: Optional[str] = Field(None, description="Filter by professional title/headline substring")
    skill: Optional[str] = Field(None, description="Filter by a skill substring, case-insensitive")

def _tool_search_profiles(text: Optional[str] = None, title: Optional[str] = None, skill: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search stored profiles by text, title, or skill. Returns a list of matching profiles."""
    return DBI.search(text=text, title=title, skill=skill)

SearchProfiles = StructuredTool.from_function(
    name="search_profiles",
    description=(
        "Search across stored profiles. Use for employer queries like 'show me ML engineers with NLP'."
    ),
    func=_tool_search_profiles,
    args_schema=SearchArgs,
)

class GetArgs(BaseModel):
    name: str = Field(..., description="Exact full name of the profile to fetch")

def _tool_get_profile(name: str) -> Optional[Dict[str, Any]]:
    """Fetch a single profile by exact name. Returns null if not found."""
    return DBI.get_profile(name)

GetProfile = StructuredTool.from_function(
    name="get_profile", description="Get one profile by exact name.", func=_tool_get_profile, args_schema=GetArgs
)

TOOLS_EMPLOYEE = [UpsertProfile, GetProfile]
TOOLS_EMPLOYER = [SearchProfiles, GetProfile]

# -----------------------------
# Prompts (role-based + few-shot)
# -----------------------------
EMPLOYEE_SYSTEM = (
    "You are a world-class Career Profile Builder. "
    "Your goal is to co-create a crisp, compelling professional profile.\n"
    "Principles:\n"
    "- Ask one thoughtful question at a time.\n"
    "- Be specific: probe for achievements, scope, metrics, and impact.\n"
    "- Use concise, professional language.\n"
    "- Regularly call save_profile to persist partial updates.\n"
    "- Stop asking when the profile feels complete; present a neat summary and save.\n"
    "- If the user already entered a name, assume they are the profile owner.\n"
)

EMPLOYEE_FEWSHOT: List[Any] = [
    SystemMessage(content=(
        "Few-shot: Ask focused questions like 'What measurable outcome are you proud of?' "
        "and 'Which tools did you use?' Avoid broad or repetitive prompts."
    )),
]

EMPLOYER_SYSTEM = (
    "You are a Recruiter Concierge. \n"
    "- Interpret employer queries, then search_profiles or get_profile as needed.\n"
    "- Summarize top candidates with headlines, top skills, notable achievements.\n"
    "- Keep replies brief and scannable; use bullet points.\n"
    "- If no matches, suggest broader or alternative search terms.\n"
)

# -----------------------------
# LangGraph State & Nodes
# -----------------------------
class GraphState(TypedDict):
    messages: List[Any]
    mode: str  # 'employee' | 'employer'

memory = MemorySaver()


def build_llm(tools: List[StructuredTool]):
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)
    return llm.bind_tools(tools)


def employee_node(state: GraphState) -> GraphState:
    tools = TOOLS_EMPLOYEE
    llm = build_llm(tools)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=EMPLOYEE_SYSTEM),
        *EMPLOYEE_FEWSHOT,
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))

    # Auto-persist if the model attempted a tool call
    new_messages: List[Any] = state["messages"] + [response]
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool = {t.name: t for t in tools}[call["name"]]
            tool_result = tool.invoke(call["args"])  # sync call
            new_messages += [
                AIMessage(content="(tool) Executed: {}".format(call["name"]), name=call["name"]) ,
                AIMessage(content=json.dumps(tool_result, ensure_ascii=False)[:4000], name=f"{call['name']}_result"),
            ]
    return {**state, "messages": new_messages}


def employer_node(state: GraphState) -> GraphState:
    tools = TOOLS_EMPLOYER
    llm = build_llm(tools)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=EMPLOYER_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))

    new_messages: List[Any] = state["messages"] + [response]
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool = {t.name: t for t in tools}[call["name"]]
            tool_result = tool.invoke(call["args"])  # sync call
            new_messages += [
                AIMessage(content="(tool) Executed: {}".format(call["name"]), name=call["name"]) ,
                AIMessage(content=json.dumps(tool_result, ensure_ascii=False)[:4000], name=f"{call['name']}_result"),
            ]
    return {**state, "messages": new_messages}


# Router decides which node to run (based on fixed mode)

def router(state: GraphState):
    return "employee" if state["mode"] == "employee" else "employer"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("employee", employee_node)
    g.add_node("employer", employer_node)
    g.set_entry_point(router)
    g.add_edge("employee", END)
    g.add_edge("employer", END)
    return g.compile(checkpointer=memory)


# -----------------------------
# CLI Loop
# -----------------------------

def banner(mode: str, name: Optional[str]):
    title = f"Employee Mode — Building Profile for {name}" if mode == "employee" else "Employer Mode — Talent Concierge"
    console.print(Panel.fit(title, style="bold cyan"))


def pretty_profiles(rows: List[Dict[str, Any]]):
    if not rows:
        console.print(Panel("No results.", style="red"))
        return
    for p in rows:
        t = Table(show_header=True, header_style="bold magenta")
        t.title = p.get("name", "(unnamed)")
        t.add_column("Field", style="dim", width=16)
        t.add_column("Value")
        t.add_row("Title", str(p.get("title", "")))
        t.add_row("Location", str(p.get("location", "")))
        t.add_row("Summary", str(p.get("summary", "")))
        t.add_row("Skills", ", ".join(p.get("skills", [])))
        if p.get("experiences"):
            exp_lines = []
            for e in p["experiences"]:
                head = f"{e.get('role','')} @ {e.get('company','')} ({e.get('start','?')}–{e.get('end','?')})"
                bullets = "\n  - " + "\n  - ".join(e.get("achievements", [])[:5]) if e.get("achievements") else ""
                exp_lines.append(head + bullets)
            t.add_row("Experience", "\n".join(exp_lines))
        if p.get("education"):
            t.add_row("Education", "\n".join(p["education"]))
        if p.get("links"):
            t.add_row("Links", "\n".join(p["links"]))
        console.print(t)


def run_cli(mode: str, name: Optional[str]):
    app = build_graph()
    banner(mode, name)

    # Prime the conversation per mode
    if mode == "employee":
        if not name:
            console.print("[bold yellow]Tip:[/] You can pass --name to prefill. We'll ask for it anyway.")
        system_kickoff = (
            "Let's create your professional profile. We'll go step by step. "
            "I'll ask one question at a time and save as we go."
        )
        msgs: List[Any] = [HumanMessage(content=system_kickoff)]
        if name:
            msgs.append(HumanMessage(content=f"My name is {name}."))
    else:
        msgs = [HumanMessage(content=(
            "You're assisting an employer. You can ask me for filters (role, skills, location) or search directly."
        ))]

    while True:
        # One step through the graph (single LLM turn + optional tool call)
        state: GraphState = {"messages": msgs, "mode": mode}
        state = app.invoke(state)

        # Extract last AI message for display
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and not m.name), None)
        if last_ai:
            console.print(Panel(last_ai.content, style="green"))

        # If employer and there are tool results (search), show nicely
        tool_results = [m for m in state["messages"] if isinstance(m, AIMessage) and m.name and m.name.endswith("_result")]
        for r in tool_results[-1:]:  # show only the latest result set
            try:
                payload = json.loads(r.content)
                if isinstance(payload, list):
                    pretty_profiles(payload)
                elif isinstance(payload, dict) and payload.get("name"):
                    pretty_profiles([payload])
            except Exception:
                pass

        # Get next user input
        try:
            user = console.input("[bold cyan]\nYou:[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye! 👋")
            break
        if user.strip().lower() in {"exit", "quit"}:
            console.print("Exiting. Profiles saved to: " + DB_PATH)
            break
        msgs = state["messages"] + [HumanMessage(content=user)]


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Conversational CV — Dual Mode")
    parser.add_argument("--mode", choices=["employee", "employer"], required=True, help="Choose the journey")
    parser.add_argument("--name", type=str, default=None, help="Employee's full name (prefill)")
    args = parser.parse_args()

    # Fail fast if no API key
    if not (os.getenv("OPENAI_API_KEY")):
        console.print("[bold red]OPENAI_API_KEY[/] is not set. Please export it and retry.")
        sys.exit(1)

    run_cli(mode=args.mode, name=args
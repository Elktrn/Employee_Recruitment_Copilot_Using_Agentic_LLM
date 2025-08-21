import json
import os
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage,AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent


# Set our OpenAI API key and load it form the env variables
from dotenv import load_dotenv
load_dotenv()

class UserMode(Enum):
    EMPLOYEE = "employee"
    EMPLOYER = "employer"

class ProfileSection(Enum):
    BASIC_INFO = "basic_info"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    COMPLETE = "complete"

@dataclass
class ProfessionalProfile:
    name: str = ""
    profession: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""
    experiences: list[dict] = None
    skills: list[str] = None
    education: list[dict] = None
    created_at: str = ""
    
    def __post_init__(self):
        if self.experiences is None:
            self.experiences = []
        if self.skills is None:
            self.skills = []
        if self.education is None:
            self.education = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
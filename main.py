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
# in VS code add the openai api key to .env file and then lead_dotenv() as below
# if you use any other environment make sure that openai api key is added to the environmental variables
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

class ProfileDatabase:
    """Persistent JSON-based database for storing profiles"""

    def __init__(self, filepath: str = "profiles.json"):
        self.filepath = filepath
        self.profiles: dict[str, ProfessionalProfile] = self._load_from_file()

    def _load_from_file(self) -> dict[str, ProfessionalProfile]:
        """Load profiles from JSON file if exists, else empty dict"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        name: ProfessionalProfile(**profile)
                        for name, profile in data.items()
                    }
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_to_file(self):
        """Persist all profiles to JSON file"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(
                {name: profile.__dict__ for name, profile in self.profiles.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save_profile(self, profile: ProfessionalProfile) -> bool:
        """Save a profile to the database"""
        if profile.name:
            self.profiles[profile.name.lower()] = profile
            self._save_to_file()
            return True
        return False

    def get_profile(self, name: str) -> ProfessionalProfile | None:
        return self.profiles.get(name.lower())

    def list_profiles(self) -> list[str]:
        return list(self.profiles.keys())

    def search_profiles(self, query: str) -> list[ProfessionalProfile]:
        query = query.lower()
        results = []
        for profile in self.profiles.values():
            if (
                query in profile.name.lower()
                or query in profile.profession.lower()
                or any(query in skill.lower() for skill in profile.skills)
            ):
                results.append(profile)
        return results

class CVBuilderAgent:
    """Main agent class for building professional profiles"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.database = ProfileDatabase()
        self.current_profile = ProfessionalProfile()
        self.current_section = ProfileSection.BASIC_INFO
        self.memory = ConversationBufferMemory(return_messages=True)
        self.mode = UserMode.EMPLOYEE
        
        # Initialize tools and agent
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        """Setup LangChain tools for profile management"""
        
        @tool
        def update_profile_field(field: str, value: str) -> str:
            """Update a specific field in the current profile"""
            if hasattr(self.current_profile, field):
                setattr(self.current_profile, field, value)
                return f"Updated {field} to: {value}"
            return f"Field {field} not found"
        
        @tool
        def add_experience(company: str, position: str, duration: str, description: str) -> str:
            """Add work experience to the profile"""
            experience = {
                "company": company,
                "position": position,
                "duration": duration,
                "description": description
            }
            self.current_profile.experiences.append(experience)
            return f"Added experience: {position} at {company}"
        
        @tool
        def add_skills(skills_list: str) -> str:
            """Add skills to the profile (comma-separated)"""
            skills = [skill.strip() for skill in skills_list.split(",")]
            self.current_profile.skills.extend(skills)
            return f"Added skills: {', '.join(skills)}"
        
        @tool
        def add_education(institution: str, degree: str, year: str) -> str:
            """Add education to the profile"""
            education = {
                "institution": institution,
                "degree": degree,
                "year": year
            }
            self.current_profile.education.append(education)
            return f"Added education: {degree} from {institution}"
        
        @tool
        def save_current_profile() -> str:
            """Save the current profile to database"""
            if self.database.save_profile(self.current_profile):
                return f"Profile for {self.current_profile.name} saved successfully!"
            return "Failed to save profile. Please ensure name is provided."
        
        @tool
        def search_profiles_tool(query: str) -> str:
            """Search for profiles in the database"""
            results = self.database.search_profiles(query)
            if not results:
                return f"No profiles found matching '{query}'"
            
            output = f"Found {len(results)} profile(s):\n"
            for profile in results:
                output += f"f{self.get_profile_summary(profile.name)}\n"
            return output
        
        self.tools = [
            update_profile_field,
            add_experience,
            add_skills,
            add_education,
            save_current_profile,
            search_profiles_tool
        ]
    
    def _setup_agent(self):
        """Setup the LangChain agent with prompts"""
        
        employee_system_prompt = """You are a professional CV building assistant. Your role is to help users create comprehensive professional profiles through engaging conversation.

CONVERSATION FLOW:
1. Start by getting basic information (name, profession, contact details)
2. Explore work experience with thoughtful follow-up questions
3. Discuss skills and expertise areas
4. Cover education background
5. Highlight achievements and accomplishments

INTERACTION GUIDELINES:
- Ask ONE question at a time to avoid overwhelming the user
- Use follow-up questions to get richer details
- Show genuine interest in their professional journey
- Provide encouragement and positive reinforcement
- Use the available tools to structure the information properly

EXAMPLE FOLLOW-UP QUESTIONS:
- "What was your proudest achievement in that role?"
- "Can you tell me about a challenging project you led?"
- "What specific technologies or methodologies did you use?"
- "How did you measure success in that position?"

Remember to use the provided tools to save information as you gather it. Be conversational, professional, and thorough."""

        employer_system_prompt = """You are a professional recruitment assistant helping employers explore candidate profiles.

Your capabilities include:
- Searching profiles by skills, profession, or name
- Providing detailed profile summaries
- Comparing candidates
- Suggesting relevant matches

Be professional, helpful, and provide clear, actionable information to help employers make informed decisions."""

        system_prompt = employee_system_prompt if self.mode == UserMode.EMPLOYEE else employer_system_prompt
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def switch_mode(self, mode: UserMode):
        """Switch between Employee and Employer modes"""
        self.mode = mode
        if mode == UserMode.EMPLOYEE:
            self.current_profile = ProfessionalProfile()
        self.memory.clear()
        self._setup_agent()
        
        welcome_msg = {
            UserMode.EMPLOYEE: "👋 Welcome to the Professional Profile Builder! I'm here to help you create an impressive CV through our conversation. Let's start with your name - what should I call you?",
            UserMode.EMPLOYER: "👋 Welcome to the Employer Portal! I can help you search and explore professional profiles. What kind of candidate or skills are you looking for today?"
        }
        
        return welcome_msg[mode]
    
    def chat(self, user_input: str) -> str:
        """Process user input and return agent response"""
        try:
            # Get chat history
            chat_history = self.memory.chat_memory.messages
            
            # Run the agent
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Update memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response["output"])
            
            return response["output"]
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Could you please try rephrasing your message?"
    
    def get_profile_summary(self, name: str = None) -> str:
        """Get a formatted summary of a profile"""
        profile = self.current_profile if not name else self.database.get_profile(name)
        
        if not profile or not profile.name:
            return "No profile found or profile incomplete."
        
        summary = f"""
📋 PROFESSIONAL PROFILE: {profile.name.upper()}

👤 Basic Information:
   • Profession: {profile.profession}
   • Email: {profile.email}
   • Phone: {profile.phone}
   • Location: {profile.location}

📝 Professional Summary:
   {profile.summary}

💼 Work Experience:
"""
        for exp in profile.experiences:
            summary += f"   • {exp['position']} at {exp['company']} ({exp['duration']})\n"
            summary += f"     {exp['description']}\n"
        
        summary += f"\n🛠️ Skills: {', '.join(profile.skills)}\n"
        
        summary += "\n🎓 Education:\n"
        for edu in profile.education:
            summary += f"   • {edu['degree']} - {edu['institution']} ({edu['year']})\n"
        
        return summary

def main():
    """Main application loop"""
    print("🚀 Professional Profile Builder")
    print("=" * 50)
    
    # Initialize the agent (you'll need to set your OpenAI API key)
    try:
        agent = CVBuilderAgent()
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    while True:
        print("\n📋 SELECT MODE:")
        print("1. Employee Mode (Build your profile)")
        print("2. Employer Mode (Search profiles)")
        print("3. View Profile Summary")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            print("🔧 EMPLOYEE MODE")
            print("="*50)
            welcome_msg = agent.switch_mode(UserMode.EMPLOYEE)
            print(f"\n🤖 Agent: {welcome_msg}")
            
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                
                response = agent.chat(user_input)
                print(f"\n🤖 Agent: {response}")
        
        elif choice == "2":
            print("\n" + "="*50)
            print("🏢 EMPLOYER MODE")
            print("="*50)
            welcome_msg = agent.switch_mode(UserMode.EMPLOYER)
            print(f"\n🤖 Agent: {welcome_msg}")
            
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                
                response = agent.chat(user_input)
                print(f"\n🤖 Agent: {response}")
        
        elif choice == "3":
            print("\n" + "="*50)
            print("📊 PROFILE SUMMARY")
            print("="*50)
            
            profiles = agent.database.list_profiles()
            if not profiles:
                print("No profiles available.")
            else:
                print("Available profiles:")
                for i, name in enumerate(profiles, 1):
                    print(f"{i}. {name.title()}")
                
                try:
                    selection = int(input("\nSelect profile number: ")) - 1
                    if 0 <= selection < len(profiles):
                        summary = agent.get_profile_summary(profiles[selection])
                        print(summary)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
        
        elif choice == "4":
            print("\n👋 Thank you for using Professional Profile Builder!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()
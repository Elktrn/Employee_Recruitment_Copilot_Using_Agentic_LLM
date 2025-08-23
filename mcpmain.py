import os
import json
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from mcp import Client, Tool
# Set our OpenAI API key and load it form the env variables
# in VS code add the openai api key to .env file and then lead_dotenv() as below
# if you use any other environment make sure that openai api key is added to the environmental variables
from dotenv import load_dotenv
load_dotenv()

# ---------------------- ENUMS ---------------------- #
class UserMode(Enum):
    EMPLOYEE = "employee"
    EMPLOYER = "employer"

class ProfileSection(Enum):
    BASIC_INFO = "basic_info"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    COMPLETE = "complete"

# ---------------------- DATA MODEL ---------------------- #
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

# ---------------------- PERSISTENCE ---------------------- #
class ProfileDatabase:
    def __init__(self, filepath: str = "profiles.json"):
        self.filepath = filepath
        self.profiles: dict[str, ProfessionalProfile] = self._load_from_file()

    def _load_from_file(self) -> dict[str, ProfessionalProfile]:
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
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(
                {name: profile.__dict__ for name, profile in self.profiles.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save_profile(self, profile: ProfessionalProfile) -> bool:
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

# ---------------------- MAIN AGENT ---------------------- #
class CVBuilderAgent:
    def __init__(self):
        self.client = Client(model=os.getenv("OPENAI_MODEL", "gpt-4"))
        self.database = ProfileDatabase()
        self.current_profile = ProfessionalProfile()
        self.current_section = ProfileSection.BASIC_INFO
        self.chat_history: list[dict] = []  # {role, content}
        self.mode = UserMode.EMPLOYEE
        self._setup_tools()
        self._setup_prompts()

    def _setup_tools(self):
        self.client.register_tool(Tool(
            name="update_profile_field",
            description="Update a specific field in the current profile",
            func=lambda field, value: self._update_profile_field(field, value)
        ))

        self.client.register_tool(Tool(
            name="add_experience",
            description="Add work experience to the profile",
            func=lambda company, position, duration, description: self._add_experience(company, position, duration, description)
        ))

        self.client.register_tool(Tool(
            name="add_skills",
            description="Add skills to the profile (comma-separated)",
            func=lambda skills_list: self._add_skills(skills_list)
        ))

        self.client.register_tool(Tool(
            name="add_education",
            description="Add education to the profile",
            func=lambda institution, degree, year: self._add_education(institution, degree, year)
        ))

        self.client.register_tool(Tool(
            name="save_current_profile",
            description="Save the current profile to database",
            func=lambda: self._save_current_profile()
        ))

        self.client.register_tool(Tool(
            name="search_profiles_tool",
            description="Search for profiles in the database",
            func=lambda query: self._search_profiles_tool(query)
        ))

    def _setup_prompts(self):
        self.employee_system_prompt = """You are a professional CV building assistant. Your role is to help users create comprehensive professional profiles through engaging conversation.

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
- Use the available tools to structure the information properly"""

        self.employer_system_prompt = """You are a professional recruitment assistant helping employers explore candidate profiles.

Your capabilities include:
- Searching profiles by skills, profession, or name
- Providing detailed profile summaries
- Comparing candidates
- Suggesting relevant matches

Be professional, helpful, and provide clear, actionable information to help employers make informed decisions."""

    # ---------------------- Tool Implementations ---------------------- #
    def _update_profile_field(self, field: str, value: str) -> str:
        if hasattr(self.current_profile, field):
            setattr(self.current_profile, field, value)
            return f"Updated {field} to: {value}"
        return f"Field {field} not found"

    def _add_experience(self, company: str, position: str, duration: str, description: str) -> str:
        exp = {"company": company, "position": position, "duration": duration, "description": description}
        self.current_profile.experiences.append(exp)
        return f"Added experience: {position} at {company}"

    def _add_skills(self, skills_list: str) -> str:
        skills = [s.strip() for s in skills_list.split(",")]
        self.current_profile.skills.extend(skills)
        return f"Added skills: {', '.join(skills)}"

    def _add_education(self, institution: str, degree: str, year: str) -> str:
        edu = {"institution": institution, "degree": degree, "year": year}
        self.current_profile.education.append(edu)
        return f"Added education: {degree} from {institution}"

    def _save_current_profile(self) -> str:
        if self.database.save_profile(self.current_profile):
            return f"Profile for {self.current_profile.name} saved successfully!"
        return "Failed to save profile. Please ensure name is provided."

    def _search_profiles_tool(self, query: str) -> str:
        results = self.database.search_profiles(query)
        if not results:
            return f"No profiles found matching '{query}'"
        output = f"Found {len(results)} profile(s):\n"
        for profile in results:
            output += f"{self.get_profile_summary(profile.name)}\n"
        return output

    # ---------------------- Conversation ---------------------- #
    def switch_mode(self, mode: UserMode):
        self.mode = mode
        if mode == UserMode.EMPLOYEE:
            self.current_profile = ProfessionalProfile()
        self.chat_history.clear()
        return {
            UserMode.EMPLOYEE: "👋 Welcome to the Professional Profile Builder! Let's start with your name - what should I call you?",
            UserMode.EMPLOYER: "👋 Welcome to the Employer Portal! What kind of candidate or skills are you looking for today?"
        }[mode]

    def chat(self, user_input: str) -> str:
        self.chat_history.append({"role": "user", "content": user_input})
        try:
            system_prompt = self.employee_system_prompt if self.mode == UserMode.EMPLOYEE else self.employer_system_prompt
            messages = [{"role": "system", "content": system_prompt}] + self.chat_history
            response = self.client.chat(
                messages=messages,
                tools=True  # enables registered tools
            )
            reply = response["content"]
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"I encountered an error: {str(e)}"

    def get_profile_summary(self, name: str = None) -> str:
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
main()
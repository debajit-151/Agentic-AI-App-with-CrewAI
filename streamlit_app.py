from crewai import Agent, Task, Crew, LLM 
from crewai_tools import SerperDevTool
import streamlit_app as st
import os

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", os.getenv("SERPER_API_KEY"))

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY not found. Please set it in Streamlit Secrets or environment.")
if not SERPER_API_KEY:
    st.warning("⚠️ SERPER_API_KEY not found. Please set it in Streamlit Secrets or environment.")

st.set_page_config(page_title="CrewAI - Content Generation", page_icon="🧠", layout="wide")

st.title("Content Generation with CrewAI")
st.markdown("Welcome to the CrewAI demo! This app demonstrates how CrewAI agents can collaborate to generate content efficiently.")

with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Topic", "Medical Industry using Generative AI", placeholder="Enter the topic for content generation")
    st.markdown("### LLM Configuration")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    st.markdown("### Search Tool Configuration")
    n = st.slider("Number of Search Results", min_value=1, max_value=20, value=10, step=1)
    generate_button = st.button("Generate Content", type="primary")

def generate_content(topic):
    llm = LLM(model="gpt-4o-mini", provider="openai", api_key=OPENAI_API_KEY, temperature=temperature)
    search_tool = SerperDevTool(n=n, api_key=SERPER_API_KEY)

    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research on {topic}",
        backstory="You are a Senior Research Analyst with 5 years of experience in AI and the medical industry.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    content_writer = Agent(
        role="Content Writer",
        goal=f"Write a detailed article on {topic}",
        backstory="You are an experienced Content Writer with expertise in AI and healthcare topics.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    research_task = Task(
        description=f"Research on {topic}",
        expected_output="A detailed summary of the current trends and applications of Generative AI in the Medical Industry",
        agent=senior_research_analyst
    )

    writing_task = Task(
        description=f"Write a detailed article on {topic}",
        expected_output="A well-researched article on the applications of Generative AI in the Medical Industry",
        agent=content_writer
    )

    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    return crew.kickoff(inputs={"topic": topic})

if generate_button:
    with st.spinner("Generating content..."):
        try:
            result = generate_content(topic)
            st.markdown("### Content Generation Result")
            st.markdown(result)
            st.download_button(
                label="Download Content",
                data=result,
                file_name=f"{topic.replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.markdown("Built with ❤️ by [CrewAI](https://crewai.com)")

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from tools.custom_tool import RAGTool
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.7,
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ClaimsExaminer():
    """ClaimsExaminer crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def query_structuring_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['query_structuring_agent'], # type: ignore[index]
            verbose=True,
            llm=llm  # Use the LLM instance for this agent
        )

    @agent
    def approver_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['approver_agent'],# type: ignore[index]
            tools=[RAGTool()],  # Register the RAGTool with the agent
            verbose=True,
            llm=llm  # Use the LLM instance for this agent
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def structuring_task(self) -> Task:
        return Task(
            config=self.tasks_config['structuring_task'], # type: ignore[index]
        )

    @task
    def approval_task(self) -> Task:
        return Task(
            config=self.tasks_config['approval_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Hackrx crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
    
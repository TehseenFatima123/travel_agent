from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from travel_tools import get_flights, suggest_hotel
load_dotenv()
client = AsyncOpenAI(
    api_key = os.getenv("GEMINI-API-KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(model= "gemini-2.0-flash", openai_client = client)
config = RunConfig(model=model, tracing_disabled=True)
destination_agent = Agent(
    name = "Destination_Agent",
    instructions = "You give suggestion of destination according to customer's mood and interest.",
    model=model

)
booking_agent = Agent(
    name = "Booking_Agent",
    instructions = "You suggest available flights in customer's budget using tools.",
    model=model,
    tools= [get_flights, suggest_hotel]
)
explore_agent = Agent(
    name = "Explorer_Agent",
    instructions = "You suggest different food and places to explore on the destination.",
    model=model,
)
def main():
    print("AI Trave Agent\n")
    mood = input ("What sort of place do you want to go, relaxing/adventurouds/historical/etc.?")
    result1 = Runner.run_sync(destination_agent, mood, run_config=config)
    dest = result1.final_output.strip()
    print ("\n Destination Suggested; ", dest)
    result2 = Runner.run_sync(booking_agent, dest, run_config=config)
    print ("\n Bookinginfo;", result2. final_output)
if __name__ == "__main__":
    main()

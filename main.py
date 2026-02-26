"""Interactive CLI for the multi-agent system."""

import asyncio

import logfire
from dotenv import load_dotenv

load_dotenv()

# Print agent traces to the terminal (no Logfire account required).
# Every tool call, argument, return value, and LLM response is shown as an
# indented span so you can follow the agent's full reasoning chain.
logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(
        verbose=True,
        span_style="indented",
        include_timestamps=False,
    ),
)
logfire.instrument_pydantic_ai()

EXAMPLES = [
    "For entity_42, which scenario yields the highest effect?",
    "What action should we consider for entity_1?",
    "Compare entity_5 and entity_12 across all actions",
    "Plot entity_7's cost vs effect across all scenarios",
    "Show me a bar chart for entity_20",
    "Generate a global heatmap of all entities",
    "Generate a 1-2 page report summarizing the dataset",
    "What are the top 10 entities by mean intervention effect?",
]

BANNER = """\
╔══════════════════════════════════════════════════════╗
║            Multi-Agent Analysis System               ║
║   Analyze · Visualize · Report  —  powered by AI    
 ║
╚══════════════════════════════════════════════════════╝

Example queries:
"""


def _display(result_output) -> None:  # type: ignore[no-untyped-def]
    """Pretty-print an OrchestratorResponse."""
    print(f"\nAgent: {result_output.answer}")
    if result_output.plot_paths:
        print("\nPlots saved:")
        for p in result_output.plot_paths:
            print(f"  • {p}")
    if result_output.report_path:
        print(f"\nReport saved: {result_output.report_path}")
    print()


async def main() -> None:
    # Import here so .env is loaded before pydantic-ai initialises the model
    from agents.orchestrator import get_orchestrator

    orchestrator, deps = get_orchestrator()

    print(BANNER)
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  {i}. {ex}")
    print("\nType 'quit' or 'exit' to stop.\n")

    history: list = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            result = await orchestrator.run(query, deps=deps, message_history=history)
            history = result.all_messages()
            _display(result.output)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())

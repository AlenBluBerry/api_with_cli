import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from rich.prompt import Prompt
from rich.rule import Rule


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("QWEN_API_KEY")

if not API_KEY:
    raise ValueError(
        "QWEN_API_KEY is not set.\n"
        "Make sure your .env file contains:\n"
        "QWEN_API_KEY=your_key_here"
    )

BASE_URL = "https://rimless-operator-abacus.ngrok-free.dev/v1"
MODEL = "qwen2.5-0.5b"

MAX_TOKENS = 500
TEMPERATURE = 0.7


# ============================================================
# Rich console
# ============================================================

console = Console()


# ============================================================
# OpenAI-compatible client
# ============================================================

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


# ============================================================
# Conversation memory
# ============================================================

messages = []


# ============================================================
# UI
# ============================================================

def show_banner():
    console.clear()

    banner = Text()
    banner.append("QWEN ", style="bold cyan")
    banner.append("AI CHAT", style="bold white")

    console.print(
        Panel(
            banner,
            subtitle="OpenAI-Compatible API",
            border_style="cyan",
            padding=(1, 4)
        )
    )

    console.print(
        "[dim]Commands:[/dim] "
        "[bold]exit[/bold] quit  •  "
        "[bold]clear[/bold] reset conversation  •  "
        "[bold]multiline[/bold] multi-line input"
    )

    console.print()


# ============================================================
# Parse AI output
# ============================================================

def render_ai_response(response_text):
    """
    Detect fenced code blocks and render them using Rich Syntax.
    Everything else is rendered as Markdown.
    """

    # Matches:
    #
    # ```python
    # print("Hello")
    # ```
    #
    # ```javascript
    # console.log("Hello")
    # ```

    pattern = r"```([\w+#.-]*)\n(.*?)```"

    matches = list(re.finditer(
        pattern,
        response_text,
        re.DOTALL
    ))

    # No code blocks
    if not matches:
        console.print(
            Panel(
                Markdown(response_text),
                title="🤖 Qwen",
                border_style="green",
                padding=(1, 2)
            )
        )
        return

    current_position = 0

    for match in matches:

        # Text before code
        before_code = response_text[
            current_position:match.start()
        ].strip()

        if before_code:
            console.print(
                Markdown(before_code)
            )

        language = match.group(1).strip()
        code = match.group(2)

        if not language:
            language = "text"

        # Normalize some common language names
        language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "sh": "bash",
            "shell": "bash",
            "yml": "yaml",
            "md": "markdown",
            "html": "html",
            "css": "css",
            "json": "json",
            "java": "java",
            "cpp": "cpp",
            "c++": "cpp",
            "c": "c",
            "sql": "sql",
            "r": "r"
        }

        language = language_map.get(
            language.lower(),
            language
        )

        console.print(
            Panel(
                Syntax(
                    code.rstrip(),
                    language,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=False,
                    indent_guides=True
                ),
                title=f"📋 {language}",
                subtitle="Select/copy directly from terminal",
                border_style="blue",
                padding=(1, 1)
            )
        )

        current_position = match.end()

    # Remaining text
    remaining = response_text[current_position:].strip()

    if remaining:
        console.print(
            Markdown(remaining)
        )


# ============================================================
# User input
# ============================================================

def get_multiline_input():
    """
    Allows the user to enter multiple lines.

    Finish the input by typing:
    END
    """

    console.print(
        Panel(
            "Enter your prompt below.\n"
            "Type [bold cyan]END[/bold cyan] on a new line when finished.",
            title="Multiline Input",
            border_style="yellow"
        )
    )

    lines = []

    while True:
        try:
            line = input()

            if line.strip() == "END":
                break

            lines.append(line)

        except EOFError:
            break

    return "\n".join(lines).strip()


# ============================================================
# Send message
# ============================================================

def ask_qwen(user_input):

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:

        with console.status(
            "[bold cyan]Qwen is thinking...[/bold cyan]",
            spinner="dots"
        ):

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

        answer = response.choices[0].message.content

        # Save AI response for conversation memory
        messages.append({
            "role": "assistant",
            "content": answer
        })

        console.print()

        render_ai_response(answer)

        console.print()

    except Exception as error:

        # Remove failed user message
        if messages and messages[-1]["role"] == "user":
            messages.pop()

        console.print(
            Panel(
                str(error),
                title="❌ API Error",
                border_style="red"
            )
        )


# ============================================================
# Main application
# ============================================================

def main():

    show_banner()

    while True:

        try:

            user_input = Prompt.ask(
                "[bold cyan]You[/bold cyan]"
            ).strip()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        except EOFError:
            break

        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if not user_input:
            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if user_input.lower() in ("exit", "quit"):
            console.print()
            console.print(
                "[bold cyan]Goodbye! 👋[/bold cyan]"
            )
            break

        # ----------------------------------------------------
        # Clear conversation
        # ----------------------------------------------------

        if user_input.lower() == "clear":

            messages.clear()

            console.clear()
            show_banner()

            console.print(
                "[green]✓ Conversation cleared.[/green]\n"
            )

            continue

        # ----------------------------------------------------
        # Multiline mode
        # ----------------------------------------------------

        if user_input.lower() == "multiline":

            user_input = get_multiline_input()

            if not user_input:
                continue

        # ----------------------------------------------------
        # Display user's message
        # ----------------------------------------------------

        console.print()

        console.print(
            Panel(
                Markdown(user_input),
                title="👤 You",
                border_style="cyan",
                padding=(1, 2)
            )
        )

        console.print()

        # ----------------------------------------------------
        # Send to Qwen
        # ----------------------------------------------------

        ask_qwen(user_input)

        console.print(Rule(style="dim"))


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    main()

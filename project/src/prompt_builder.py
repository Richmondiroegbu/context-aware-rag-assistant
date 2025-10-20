from typing import List, Dict, Optional, Union
from textwrap import dedent



class PromptBuilder:
    """
    A dynamic Prompt Builder for LLMs.

    - No hardcoded templates or reasoning modes.
    - User defines only relevant sections such as system prompt, reasoning instructions, roles, goals, etc.
    - Produces both a structured dictionary and a formatted text prompt.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """
        Initialize a new PromptBuilder instance.

        Args:
            system_prompt (Optional[str]): The system-level instruction or role description for the LLM.
        """
        self._system_prompt: str = dedent(system_prompt).strip() if system_prompt else "You are a helpful AI assistant."
        self.reasoning_instructions: Optional[str] = None

        # Containers for prompt sections
        self.roles: Dict[str, str] = {}
        self.goals: List[str] = []
        self.instructions: List[str] = []
        self.constraints: List[str] = []
        self.tones: List[str] = []




    def system_prompt(self, text: str) -> "PromptBuilder":
        """Set or update the system prompt.
        Args:
            text (str): The system prompt text.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        self._system_prompt = dedent(text).strip()
        return self


    def set_reasoning_instructions(self, instructions: str) -> "PromptBuilder":
        """Set reasoning-related guidance for the LLM.
        Args:
            instructions (str): The reasoning instructions text.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        self.reasoning_instructions = dedent(instructions).strip()
        return self
    


    def add_role(self, name: str, description: str) -> "PromptBuilder":
        """Add a role definition for the model to adopt.
        Args:
            name (str): The role name (e.g., "System", "User").
            description (str): A brief description of the role's purpose.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        self.roles[name.strip()] = dedent(description).strip()
        return self
    


    def add_goal(self, goal: Union[str, List[str]]) -> "PromptBuilder":
        """Add one or multiple high-level goals.
        Args:
            goal (Union[str, List[str]]): A single goal string or a list of goal strings.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """

        goals = goal if isinstance(goal, list) else [goal]
        self.goals.extend([dedent(g).strip() for g in goals])
        return self



    def add_instruction(self, instruction: Union[str, List[str]]) -> "PromptBuilder":
        """Add one or multiple specific task instructions.
        Args:
            instruction (Union[str, List[str]]): A single instruction string or a list of instruction strings.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        instructions = instruction if isinstance(instruction, list) else [instruction]
        self.instructions.extend([dedent(i).strip() for i in instructions])
        return self



    def add_constraint(self, constraint: Union[str, List[str]]) -> "PromptBuilder":
        """Add one or multiple operational constraints.
        Args:
            constraint (Union[str, List[str]]): A single constraint string or a list of constraint strings.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        constraints = constraint if isinstance(constraint, list) else [constraint]
        self.constraints.extend([dedent(c).strip() for c in constraints])
        return self




    def add_tone(self, tone: Union[str, List[str]]) -> "PromptBuilder":
        """Add one or multiple desired response tones.
        Args:
            tone (Union[str, List[str]]): A single tone string or a list of tone strings.
        Returns:
            PromptBuilder: The current instance for method chaining.
        """
        tones = tone if isinstance(tone, list) else [tone]
        self.tones.extend([dedent(t).strip() for t in tones])
        return self

    
    
    def build(self) -> Dict[str, Union[str, Dict, List, None]]:
        """
        Build the final structured dictionary representation of the prompt.

        Returns:
            Dict[str, Union[str, Dict, List, None]]: A dictionary containing all prompt sections.
        """
        return {
            "system_prompt": self._system_prompt,
            "reasoning_instructions": self.reasoning_instructions,
            "roles": self.roles,
            "goals": self.goals,
            "instructions": self.instructions,
            "constraints": self.constraints,
            "tones": self.tones,
            "full_prompt": self._compose_full_prompt(),
        }

    def _compose_full_prompt(self) -> str:
        """
        Combine all prompt parts into a formatted human-readable string.

        Returns:
            str: The formatted prompt string.
        """
        parts: List[str] = [f"System: {self._system_prompt}"]

        def add_section(header: str, lines: List[str]) -> None:
            if lines:
                parts.append(f"\n{header}")
                for line in lines:
                    parts.append(f"- {line}")

        if self.roles:
            parts.append("\nRoles:")
            for name, desc in self.roles.items():
                parts.append(f"- {name}: {desc}")

        add_section("Goals:", self.goals)
        add_section("Instructions:", self.instructions)
        add_section("Constraints:", self.constraints)
        add_section("Tone:", self.tones)

        if self.reasoning_instructions:
            parts.append("\nReasoning Instructions:")
            parts.append(self.reasoning_instructions)

        return "\n".join(parts).strip()

    def __repr__(self) -> str:
        """Return a concise representation of the current builder state."""
        return f"<PromptBuilder system_prompt={bool(self._system_prompt)}, reasoning_instructions={bool(self.reasoning_instructions)}>"
    
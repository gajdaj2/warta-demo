from typing import Literal

import dspy

#lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
lm = dspy.LM('openai/gpt-4o')
dspy.configure(lm=lm)

class PromptOptimalization(dspy.Signature):
    """Optimize the prompt for given task and score the prompt before and after optimization """
    prompt = dspy.InputField(desc="Prompt który musi być zoptymalizowany pod pytania z bazy wektorowej i języka prawniczego")
    prompt_task = dspy.InputField(desc="Zadanie dla zoptymalizowanego promptu")
    old_prompt_score : Literal[1, 2, 3, 4, 5] = dspy.OutputField(desc="Skoring starego promptu")
    optimized_prompt = dspy.OutputField(desc="Nowy zoptymalizowany prompt zwiększający skuteczność")
    new_prompt_score : Literal[1, 2, 3, 4, 5] = dspy.OutputField(desc="Skoring nowego promptu")
    what_was_changed = dspy.OutputField(desc="What was changed in the optimized prompt")


#result = dspy.Predict(PromptOptimalization)




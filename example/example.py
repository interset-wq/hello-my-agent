import os

from dotenv import load_dotenv
load_dotenv()

from utils.core import LLMChat, ToolExecutor


# llm = LLMChat(
#     model=os.getenv("MODEL"),
#     api_key=os.getenv("API_KEY"),
#     base_url=os.getenv("BASE_URL"),
# )
#
# messages = [{"role": "user", "content": "用一句话介绍 Python。"}]
#
# print("=== invoke ===")
# content, reasoning = llm.chat(messages, include_reasoning=True)
# print("思考:", reasoning)
# print("补全:", content)
#
# print("\n=== stream ===")
# current = None
# for kind, text in llm.stream_chat(messages, include_reasoning=True):
#     if kind != current:
#         label = "思考" if kind == "reasoning" else "补全"
#         print(f"\n{label}: ", end="", flush=True)
#         current = kind
#     print(text, end="", flush=True)
# print()

tool_exe = ToolExecutor()

def hello(name, place):
    print(f"Hello, {name}! Welcome to {place}!")

def hi(s):
    pass

def ok(s):
    pass

tool_exe.register(name='hello', description='print hello {name}', func=hello)
tool_exe.register(name='hi', description='print hi', func=hi)
tool_exe.register(name='ok', description='print ok', func=ok)

print('=== tool list ===')
print(tool_exe.tools)
print()

print('=== description ===')
print(tool_exe.describe_tools())
print()

print('=== run ===')
tool_exe.run('hello', 'world', 'China')
from functools import partial
from typing import Any, Optional, Dict, List
from uuid import uuid4

import asyncio
import aiohttp
import json

from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .request_processor import RequestProcessor
from .code_judge_utils import run_tool_calls_on_server_async


class SimJupyterTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        run_jupyter_tool_calls_on_server_async = partial(
            run_tool_calls_on_server_async,
            generate_tool_call_code=generate_tool_call_code,
            generate_tool_call_input=generate_tool_call_input
        )
        tool_connector = aiohttp.TCPConnector(limit=self.config["request_processor_concurrency"], force_close=True, enable_cleanup_closed=True)
        tool_timeout = aiohttp.ClientTimeout(total=60)
        tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
        self.request_processor = RequestProcessor(
            batch_size=32,
            batch_timeout_seconds=60,
            session=tool_session,
            concurrency=32,
            batch_submit_func=run_jupyter_tool_calls_on_server_async,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def _start_request_processor(self):
        if not self.request_processor._running:
            await self.request_processor.start()

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        assert "history_tool_calls" in kwargs, "history_tool_calls must be provided in kwargs"
        await self._start_request_processor()
        history_tool_calls = []
        for history_tool_call in kwargs["history_tool_calls"]:
            if history_tool_call.name == "jupyter_code":
                try:
                    arguments = json.loads(history_tool_call.arguments)
                    assert len(arguments) == 1 and "code" in arguments
                    history_tool_calls.append({
                        "name": "jupyter_code",
                        "arguments": {
                            "code": arguments["code"],
                        }
                    })
                except Exception as e:
                    pass

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "history_tool_calls": history_tool_calls,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        tool_call = {
            "name": "jupyter_code",
            "arguments": {
                "code": code,
            },
            "history_tool_calls": self._instance_dict[instance_id]["history_tool_calls"]
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


### Generate tool call code

code_template_setup = '''
import os
import base64
import sys
import ast
import traceback
from typing import Optional, Any
import linecache
from types import CodeType
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

class CodeExecutionError(Exception):
    """Custom exception for code execution errors with line information"""
    def __init__(self, original_error: Exception, code: str, line_offset: int = 0):
        self.original_error = original_error
        self.code = code
        self.line_offset = line_offset
        
        # Get error line number
        if hasattr(original_error, 'lineno'):
            self.lineno = original_error.lineno
        else:
            tb = getattr(original_error, '__traceback__', None)
            if tb:
                while tb.tb_next:
                    tb = tb.tb_next
                self.lineno = tb.tb_lineno
            else:
                self.lineno = -1
        
        # Adjust line number for code segment
        if self.lineno != -1:
            self.lineno += line_offset
        
        # Format error message
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        
        if self.lineno != -1:
            # Get the problematic line
            lines = code.splitlines()
            if 0 <= self.lineno - 1 < len(lines):
                error_line = lines[self.lineno - 1]
                # Create error message with line information
                super().__init__(f"{error_type} at line {self.lineno}: {error_msg}\\n  {error_line}")
                return
        
        super().__init__(f"{error_type}: {error_msg}")

class PersistentExecutor:
    def __init__(self):
        self.exec_globals = {
            '__name__': '__main__',
            '__file__': '<string>',
            '__builtins__': __builtins__
        }

    def split_code(self, code: str) -> tuple[str, Optional[str]]:
        """
        Intelligently split code into main body and last expression
        
        Args:
            code: The source code string
            
        Returns:
            tuple[str, Optional[str]]: (main code body, last expression if exists)
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            if not tree.body:
                return code, None
            
            # Check if the last node is a pure expression (not a call)
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # Get the line range of the last expression
                last_expr_start = last_node.lineno
                last_expr_end = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno
                
                # Split the code
                lines = code.splitlines()
                main_code = '\\n'.join(lines[:last_expr_start-1])
                last_expr = '\\n'.join(lines[last_expr_start-1:last_expr_end])
                return main_code, last_expr
        except SyntaxError as e:
            raise CodeExecutionError(e, code)
        return code, None

    def execute_code(self, code: str, replay_history_code: bool) -> None:
        """
        Execute code while maintaining persistent environment state.
        If the last line is an expression, its value will be printed to stdout.
        
        Args:
            code: The source code string to execute
            replay_history_code: If True, suppress stdout and stderr output
        """
        try:
            # Split code intelligently
            main_code, last_expr = self.split_code(code)
            
            # Set up output redirection if replay_history_code is True
            if replay_history_code:
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                stdout_context = redirect_stdout(stdout_capture)
                stderr_context = redirect_stderr(stderr_capture)
            else:
                stdout_context = redirect_stdout(sys.stdout)
                stderr_context = redirect_stderr(sys.stderr)
            
            # Execute main code body
            if main_code:
                try:
                    # Compile code to get better error line numbers
                    compiled_code = compile(main_code, '<string>', 'exec')
                    with stdout_context, stderr_context:
                        exec(compiled_code, self.exec_globals)
                except Exception as e:
                    raise CodeExecutionError(e, main_code)
            
            # If there's a last expression, try to evaluate and print it
            if last_expr:
                try:
                    # Compile expression to get better error line numbers
                    compiled_expr = compile(last_expr, '<string>', 'eval')
                    with stdout_context, stderr_context:
                        last_value = eval(compiled_expr, self.exec_globals)
                    
                    # Only print the result if not in replay mode
                    if last_value is not None and not replay_history_code:
                        print(repr(last_value), file=sys.stdout)
                except Exception as e:
                    # Try executing as statement if evaluation fails
                    try:
                        compiled_stmt = compile(last_expr, '<string>', 'exec')
                        with stdout_context, stderr_context:
                            exec(compiled_stmt, self.exec_globals)
                    except Exception as e:
                        # Calculate line offset for the last expression
                        line_offset = len(main_code.splitlines()) if main_code else 0
                        raise CodeExecutionError(e, last_expr, line_offset)
                    
        except Exception as e:
            if replay_history_code:
                return
            if isinstance(e, CodeExecutionError):
                print(str(e), file=sys.stderr)
            else:
                traceback.print_exc(file=sys.stderr)
            os._exit(1)
            return

persistent_executor = PersistentExecutor()
'''

code_template_exec = '''
code_to_execute = base64.b64decode("{}".encode()).decode()
persistent_executor.execute_code(code_to_execute, replay_history_code={})
'''

def combine_code_template(code_to_execute: str, history_code_to_execute: Optional[List[str]] = None) -> str:
    history_code_to_execute = history_code_to_execute or []
    final_code = code_template_setup
    for history_code in history_code_to_execute:
        final_code += code_template_exec.format(history_code, "True")
    final_code += code_template_exec.format(code_to_execute, "False")
    return final_code


def generate_tool_call_code(tool_call: Dict) -> str:
    import base64

    def jupyter_code_gencode(json_format_data: Dict) -> str:
        code_to_execute = base64.b64encode(json_format_data["arguments"]["code"].encode()).decode()
        history_code_to_execute = [
            base64.b64encode(tool_call_json["arguments"]["code"].encode()).decode()
            for tool_call_json in json_format_data.get("history_tool_calls", []) if tool_call_json["name"] == "jupyter_code"
        ]
        return combine_code_template(code_to_execute, history_code_to_execute)

    if tool_call["name"] == "jupyter_code":
        return jupyter_code_gencode(tool_call)
    else:
        raise ValueError(f"Unsupported tool call name: {tool_call['name']}")


def generate_tool_call_input(tool_call: Dict) -> str:
    if tool_call["name"] == "jupyter_code":
        return None
    else:
        raise ValueError(f"Unsupported tool call name: {tool_call['name']}")

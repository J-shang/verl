import json
import aiohttp
import asyncio
import traceback
import os
import datetime

from typing import Dict, List, Literal, Callable

# Global variable to store the path for failed submissions
_failed_submissions_path = os.path.expanduser("~")


def set_failed_submissions_path(path: str):
    """
    Set the path where failed submissions will be saved.

    Args:
        path: The directory path to save failed submissions
    """
    global _failed_submissions_path
    _failed_submissions_path = os.path.expanduser(path)
    # Create directory if it doesn't exist
    os.makedirs(_failed_submissions_path, exist_ok=True)
    print(f"Failed submissions will be saved to: {_failed_submissions_path}")


def get_failed_submissions_path() -> str:
    """
    Get the current path where failed submissions will be saved.

    Returns:
        The current path for saving failed submissions
    """
    return _failed_submissions_path


async def call_long_batch(
                        url: str,
                        submissions: List[Dict],
                        session: aiohttp.ClientSession,
                        max_retries: int = 4,
                        backoff_factor: float = 0.5):

    sub_num = len(submissions)
    results = [None] * sub_num
    sub_ids = list(range(sub_num))
    attempt_count = 0
    while submissions and attempt_count < max_retries:
        attempt_count += 1
        try:
            data = {
                "type": "batch",
                "submissions": submissions
            }
            queue_timeouts = []
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()
                for sub_id, result in zip(sub_ids, response_json['results']):
                    if result['reason'] != 'queue_timeout':
                        results[sub_id] = result
                    else:
                        queue_timeouts.append((sub_id, submissions[sub_id]))
            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]
        except aiohttp.ClientResponseError as e:
            print(f"Attempt {attempt_count}: Server responded with {e.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {attempt_count}: Caught {type(e).__name__}: {repr(e)}")
        except Exception as e:
            print(f"run_tool_calls_on_server_async Error: {e}")
            traceback.print_exc()
        finally:
            await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))

    # Save failed submissions to file if any remain after max retries
    if submissions:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_file = os.path.join(_failed_submissions_path, f"failed_submissions_{timestamp}.json")

        failed_data = {
            "timestamp": timestamp,
            "url": url,
            "max_retries": max_retries,
            "failed_submissions": []
        }

        for sub_id, submission in zip(sub_ids, submissions):
            failed_data["failed_submissions"].append({
                "original_index": sub_id,
                "submission": submission
            })

        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(submissions)} failed submissions to: {failed_file}")
        except Exception as e:
            print(f"Failed to save failed submissions: {e}")

    return results


async def run_tool_calls_on_server_async(
                                    tool_calls: List,
                                    session: aiohttp.ClientSession,
                                    language: Literal["python", "cpp"] = "python",
                                    max_retries: int = 4,
                                    backoff_factor: float = 0.5,
                                    generate_tool_call_code: Callable = None,
                                    generate_tool_call_input: Callable = None):
    submissions = []
    for tool_call in tool_calls:
        submissions.append({
            "type": language,
            "solution": generate_tool_call_code(tool_call),
            "input": generate_tool_call_input(tool_call),
        })

    url = "http://localhost:8088/run/long-batch"
    results = await call_long_batch(url, submissions, session, max_retries, backoff_factor)

    if None in results:
        failed_indices = [i for i, result in enumerate(results) if result is None]
        # throw an error if any tool call failed after max retries
        if len(failed_indices) > 0:
            raise RuntimeError(f"run_tool_calls_on_server_async failed for {len(failed_indices)} tool calls after {max_retries} attempts.")
        
    for i in range(len(results)):
        if results[i]['run_success'] and results[i]['success']:
            output_parts = []
            output_parts.append('Tool call success')
            if results[i]["stdout"]:
                output_parts.append(f'stdout: {results[i]["stdout"]}')
            if results[i]["stderr"]:
                output_parts.append(f'stderr: {results[i]["stderr"]}')
            output_parts.append(f'execution time: {results[i]["cost"]:.2f}s')
            results[i] = '\n'.join(output_parts)
        else:
            output_parts = []
            output_parts.append('Tool call failure')
            output_parts.append(f'reason: {results[i]["reason"]}')
            if results[i]["stdout"]:
                output_parts.append(f'stdout: {results[i]["stdout"]}')
            if results[i]["stderr"]:
                output_parts.append(f'stderr: {results[i]["stderr"]}')
            output_parts.append(f'execution time: {results[i]["cost"]:.2f}s')
            results[i] = '\n'.join(output_parts)

    return results

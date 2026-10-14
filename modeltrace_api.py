"""Standard-library API transports; no model tools are provided or executed."""
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def normalize_usage(usage):
    usage = usage if isinstance(usage, dict) else {}
    def count(*values):
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and value.isascii() and value.isdigit():
                return int(value)
        return None
    details = usage.get('output_tokens_details') or usage.get('completion_tokens_details') or {}
    details = details if isinstance(details, dict) else {}
    return {
        'input_tokens': count(usage.get('input_tokens'), usage.get('prompt_tokens')),
        'output_tokens': count(usage.get('output_tokens'), usage.get('completion_tokens')),
        'reasoning_output_tokens': count(details.get('reasoning_tokens')),
    }


def run_api(prompt, model, base_url, api_key, api_format='chat', timeout=180):
    parsed = urlsplit(base_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError('无效 API URL')
    base = base_url.rstrip('/')
    for suffix in ('/chat/completions', '/responses', '/messages'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if not base.endswith('/v1'):
        base += '/v1'
    headers = {'Content-Type': 'application/json', 'User-Agent': 'ModelTrace/1.0'}
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 4096}
    if api_format == 'anthropic':
        endpoint = base + '/messages'
        headers.update({'x-api-key': api_key, 'anthropic-version': '2023-06-01'})
    elif api_format == 'responses':
        endpoint = base + '/responses'
        headers['Authorization'] = 'Bearer ' + api_key
        body = {'model': model, 'input': [{'role': 'user', 'content': prompt}], 'max_output_tokens': 4096, 'stream': False}
    else:
        endpoint = base + '/chat/completions'
        headers['Authorization'] = 'Bearer ' + api_key
        body['stream'] = False
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    started = time.perf_counter()
    try:
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'API 请求失败：HTTP {error.code}（正文已隐藏）') from None
    except Exception:
        raise RuntimeError('API 连接失败、超时或响应无效（详情已隐藏）') from None
    if api_format == 'chat':
        choice = data['choices'][0]
        message = choice['message']
        if message.get('tool_calls') or message.get('function_call') or choice.get('finish_reason') != 'stop':
            raise RuntimeError('API 回答含工具调用或未完整结束')
        text = message.get('content') or ''
    elif api_format == 'anthropic':
        if data.get('stop_reason') != 'end_turn' or any(x.get('type') not in {'text', 'thinking', 'redacted_thinking'} for x in data.get('content', [])):
            raise RuntimeError('API 回答含工具调用或未完整结束')
        text = '\n'.join(x['text'] for x in data['content'] if x['type'] == 'text')
    else:
        if data.get('status') != 'completed':
            raise RuntimeError('API 回答未完整结束')
        chunks = []
        for item in data.get('output', []):
            if item.get('type') == 'reasoning':
                continue
            if item.get('type') != 'message':
                raise RuntimeError('API 回答含工具调用或未知项目')
            for part in item.get('content', []):
                if part.get('type') != 'output_text':
                    raise RuntimeError('API 回答含拒答或未知内容')
                chunks.append(part['text'])
        text = '\n'.join(chunks)
    usage = data.get('usage') or {}
    normalized = normalize_usage(usage)
    return text, normalized, time.perf_counter() - started

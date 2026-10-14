import io
import json
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch
import modeltrace_cli as cli
from modeltrace_api import run_api, normalize_usage


class CliTests(unittest.TestCase):
    def test_parallel_scheduling(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args([]).concurrency, 1)
        for flag in ['--parallel', '--concurrency', '-c']:
            self.assertEqual(parser.parse_args([flag, '3']).concurrency, 3)
        bank = json.loads(Path('data/unified_bank.json').read_text())
        text = ','.join(str((i * 79) % 355 + 1) for i in range(330))
        active = 0
        peak = 0
        lock = threading.Lock()
        barrier = threading.Barrier(3, timeout=5)
        def parallel_runner(prompt):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return text, {'output_tokens': 100}, 1.0
        with patch('sys.stderr', new=io.StringIO()):
            result = cli.test_with_codex(None, 'low', 3, 3, bank, concurrency=3, runner=parallel_runner)
        self.assertEqual(peak, 3)
        self.assertEqual(result['api_test']['received'], 3)
        threads = set()
        def serial_runner(prompt):
            threads.add(threading.get_ident())
            return text, {'output_tokens': 100}, 1.0
        with patch('sys.stderr', new=io.StringIO()):
            result = cli.test_with_codex(None, 'low', 3, 3, bank, runner=serial_runner)
        self.assertEqual(len(threads), 1)
        self.assertEqual(result['performance']['concurrency'], 1)

    def test_usage_aliases(self):
        self.assertEqual(normalize_usage({'output_tokens': None, 'completion_tokens': '123'})['output_tokens'], 123)
        for value in [None, -1, True, 'bad', 1.2]:
            self.assertIsNone(normalize_usage({'output_tokens': value})['output_tokens'])

    def test_default_remote(self):
        bank = json.loads(Path('data/unified_bank.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(bank).encode())) as call:
            cli.load_bank(cache_dir=Path(tmp))
            call.assert_called_once_with(cli.BANK_URL_DEFAULT, timeout=60)

    def test_missing_usage_performance(self):
        bank = json.loads(Path('data/unified_bank.json').read_text())
        text = ','.join(str((i * 79) % 355 + 1) for i in range(330))
        for usages, expected_tps in [([{}, {}], None), ([{'output_tokens': 100}, {}], 50)]:
            values = iter((text, usage, 2.0) for usage in usages)
            with patch('sys.stderr', new=io.StringIO()):
                result = cli.test_with_codex(None, 'low', 2, 2, bank, concurrency=1, runner=lambda prompt: next(values))
            self.assertEqual(result['performance']['tps'], expected_tps)
            self.assertEqual(result['performance']['usage_missing_runs'], 2 if expected_tps is None else 1)
            if expected_tps is None:
                self.assertIsNone(result['performance']['output_tokens'])

    def test_event_audit(self):
        events = [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '42'}},
                  {'type': 'turn.completed', 'usage': {'output_tokens': 1}}]
        self.assertEqual(cli.audit_codex_events(chr(10).join(map(json.dumps, events)))[0], '42')
        warning = {'type': 'item.completed', 'item': {'type': 'error', 'message': 'Model metadata for `claude-opus-5-5` not found. Defaulting to fallback metadata; this can degrade performance and cause issues.'}}
        self.assertEqual(cli.audit_codex_events(chr(10).join(map(json.dumps, [warning] + events)))[0], '42')
        for kind in ['command_execution', 'mcp_tool_call', 'web_search', 'file_change', 'unknown']:
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                cli.audit_codex_events(json.dumps({'type': 'item.started', 'item': {'type': kind}}))
        for raw in ['garbage', json.dumps(events[0]), json.dumps({'type': 'turn.failed'})]:
            with self.assertRaises(RuntimeError):
                cli.audit_codex_events(raw)

    def test_explicit_remote_ignores_cache(self):
        bank = json.loads(Path('data/unified_bank.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'unified_bank.local.json').write_text('{}')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(bank).encode())) as call:
                self.assertEqual(cli.load_bank('https://example.test/bank', Path(tmp))['models'], bank['models'])
                call.assert_called_once()
            with patch('urllib.request.urlopen', side_effect=TimeoutError), self.assertRaises(RuntimeError):
                cli.load_bank('https://example.test/bank', Path(tmp))

    def test_codex_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp, 'codex')
            script.write_text(chr(10).join(['#!/bin/sh', 'sleep 30', '']))
            script.chmod(0o700)
            with patch.object(cli, 'resolve_codex_executable', return_value=str(script)):
                with self.assertRaisesRegex(RuntimeError, '超时'):
                    cli.run_codex('prompt', None, 'low', timeout=0.1)

    def test_api_formats_and_tools(self):
        cases = {
            'chat': {'choices': [{'finish_reason': 'stop', 'message': {'content': '42'}}], 'usage': {'completion_tokens': 1}},
            'responses': {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '42'}]}]},
            'anthropic': {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': '42'}]},
        }
        for fmt, data in cases.items():
            with self.subTest(fmt=fmt), patch('urllib.request.OpenerDirector.open', return_value=io.BytesIO(json.dumps(data).encode())) as call:
                self.assertEqual(run_api('prompt', 'model', 'https://example.test/v1', 'secret', fmt)[0], '42')
                self.assertNotIn('tools', json.loads(call.call_args.args[0].data))
        cases['chat']['choices'][0]['message']['tool_calls'] = [{'id': 'tool'}]
        cases['responses']['output'].append({'type': 'function_call'})
        cases['anthropic']['content'].append({'type': 'tool_use'})
        for fmt, data in cases.items():
            with patch('urllib.request.OpenerDirector.open', return_value=io.BytesIO(json.dumps(data).encode())), self.assertRaises(RuntimeError):
                run_api('prompt', 'model', 'https://example.test', 'secret', fmt)

    def test_scorer_matches_main(self):
        from fingerprint import analyze_global_outputs
        bank = json.loads(Path('data/unified_bank.json').read_text())
        numbers = [((i * 79) % 355) + 1 for i in range(310)]
        outputs = [{'text': ','.join(map(str, numbers)), 'expected_count': 310}]
        expected = analyze_global_outputs(outputs, bank)
        actual = cli.analyze_global_outputs(outputs, bank)
        self.assertEqual(actual['prediction'], expected['prediction'])
        for left, right in zip(actual['results'], expected['results']):
            self.assertAlmostEqual(left['probability'], right['probability'], places=10)


if __name__ == '__main__':
    unittest.main()

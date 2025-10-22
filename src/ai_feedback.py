import os
import json
import requests
from typing import Dict, Any


API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


def call_deepseek_feedback(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Call DeepSeek R1 API to get feedback based on provided metrics.

    metrics: a dict produced by recognize_action (or similar) describing the action.
    Returns a dict with keys like 'verdict','score','advice' when successful, otherwise raises or returns fallback.
    """
    api_key = os.getenv('ARK_API_KEY')
    if not api_key:
        return {'error': 'ARK_API_KEY not set'}

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    system_prompt = (
        "你是一个专业的健身动作评估助手。输入为仪表化的动作度量数据，请返回 JSON 格式的评估，包含字段："
        "verdict(合格/不合格)、score(0-100)、advice(简短中文建议)。只返回 JSON，不要额外的文本。"
    )

    user_prompt = f"请基于以下度量数据判断是否为标准抬膝动作，并给出评分和一条简短建议：{json.dumps(metrics, ensure_ascii=False)}"

    payload = {
        "model": "deepseek-r1-250120",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    try:
        # debug: print payload summary (without API key)
        try:
            print('[AI DEBUG] Sending payload:', json.dumps(payload, ensure_ascii=False))
        except Exception:
            print('[AI DEBUG] Sending payload (could not serialize)')

        # increase timeout to 15s to allow slower responses
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        # debug: print response status and text
        try:
            print(f"[AI DEBUG] Response status: {resp.status_code}")
            print('[AI DEBUG] Response text:', resp.text)
        except Exception:
            print('[AI DEBUG] Response received but failed to print content')

        resp.raise_for_status()
        data = resp.json()
        # deepseek 返回结构中通常包含 choices -> message -> content
        content = None
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0].get('message', {}).get('content')
        if not content:
            return {'error': 'empty_response', 'raw': data}
        # 尝试解析 content 为 JSON
        try:
            parsed = json.loads(content)
            return {'ok': True, 'result': parsed}
        except Exception:
            # 如果不是严格 JSON，则尝试从文本中抽取 JSON 子串
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(content[start:end+1])
                    return {'ok': True, 'result': parsed}
                except Exception:
                    return {'error': 'parse_failed', 'raw_content': content}
            return {'error': 'no_json', 'raw_content': content}
    except Exception as e:
        return {'error': 'request_failed', 'exception': str(e)}

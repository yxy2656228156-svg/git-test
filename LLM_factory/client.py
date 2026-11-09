import os
from pathlib import Path
from dotenv import load_dotenv

# 因为现在代码在子文件夹里，所以要往上一级找 .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm_client(provider: str):
    provider = provider.upper()
    api_key = os.getenv(f"{provider}_API_KEY")
    base_url = os.getenv(f"{provider}_BASE_URL")
    model_id = os.getenv(f"{provider}_MODEL_ID")
    timeout = os.getenv(f"{provider}_TIMEOUT")

    if not api_key or not base_url or not model_id:
        raise ValueError(f"❌ 在 .env 中找不到 {provider} 的配置！")

    return model_id,api_key,base_url,timeout

def get_serpapi_client(provider: str):
    provider = provider.upper()
    api_key = os.getenv(f"{provider}_API_KEY")

    if not api_key:
        raise ValueError(f"❌ 在 .env 中找不到 {provider} 的配置！")

    return api_key
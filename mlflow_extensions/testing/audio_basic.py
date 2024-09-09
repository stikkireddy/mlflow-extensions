import base64
import typing

import requests

from mlflow_extensions.testing.helper import Modality, inject_openai_client, run_if

if typing.TYPE_CHECKING is True:
    from openai import OpenAi

    from mlflow_extensions.testing.runner import ModelContextRunner


def encode_audio_base64_from_url(audio_url: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""

    with requests.get(audio_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


@run_if(modality=Modality.AUDIO.value)
@inject_openai_client
def query_audio(
    *,
    ctx: "ModelContextRunner",
    client: "OpenAi",
    audio_data: str,
    model: str,
    modality_type: str = None,
    host: str = "0.0.0.0",  # noqa
    port: int = 9989,  # noqa
    repeat_n: int = 5,
):
    count = 0
    # repeat a few times
    while count < repeat_n:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Breakdown the content of the audio?",
                            },
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    # Any format supported by librosa is supported
                                    "url": f"data:audio/ogg;base64,{audio_data}"
                                },
                            },
                        ],
                    }
                ],
                model=model,
                max_tokens=512,
            )
            ctx.add_success(result=response.choices[0].message.content)

            from typing import List, Literal

            from pydantic import BaseModel

            class AudioExtraction(BaseModel):
                year: str
                speaker: str
                location: str
                sentiment: Literal["positive", "negative", "neutral"]
                tone: Literal["somber", "upbeat", "pessemistic"]
                summary: str

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Extract the following content from the audio: {str(AudioExtraction.schema())}?",
                            },
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    # Any format supported by librosa is supported
                                    "url": f"data:audio/ogg;base64,{audio_data}"
                                },
                            },
                        ],
                    }
                ],
                model=model,
                max_tokens=512,
                extra_body={"guided_json": AudioExtraction.schema()},
            )
            ctx.add_success(result=response.choices[0].message.content)
        except Exception as e:
            print(e)
            ctx.add_error(error_msg=str(e))
        finally:
            count += 1

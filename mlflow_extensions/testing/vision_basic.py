import typing

from mlflow_extensions.testing.helper import Modality, inject_openai_client, run_if

if typing.TYPE_CHECKING is True:
    from openai import OpenAi

    from mlflow_extensions.testing.runner import ModelContextRunner


@run_if(modality=Modality.VISION.value)
@inject_openai_client
def query_vision(
    *,
    ctx: "ModelContextRunner",
    client: "OpenAi",
    model: str,
    modality_type: str = None,
    host: str = "0.0.0.0",  # noqa
    port: int = 9989,  # noqa
    repeat_n: int = 5,
):
    count = 0
    # repeat a few times
    while count < repeat_n:
        from pydantic import BaseModel

        class ExpectedJson(BaseModel):
            outside: bool
            inside: bool
            boardwalk: bool
            grass: bool

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the images"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/73fbe271026179.5bb6e7af358b6.jpg"
                                },
                            },
                        ],
                    }
                ],
            )
            ctx.add_success(result=response.choices[0].message.content.strip())

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is the image indoors or outdoors?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                },
                            },
                        ],
                    }
                ],
                extra_body={"guided_json": ExpectedJson.schema()},
            )
            ctx.add_success(result=response.choices[0].message.content.strip())
        except Exception as e:
            print(e)
            ctx.add_error(error_msg=str(e))
        finally:
            count += 1


@run_if(modality=Modality.VISION.value)
@inject_openai_client
def query_vision_multi_input(
    *,
    ctx: "ModelContextRunner",
    client: "OpenAi",
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
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Compare the differences between the images?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/73fbe271026179.5bb6e7af358b6.jpg"
                                },
                            },
                        ],
                    }
                ],
            )
            ctx.add_success(result=response.choices[0].message.content.strip())
        except Exception as e:
            print(e)
            ctx.add_error(error_msg=str(e))
        finally:
            count += 1
